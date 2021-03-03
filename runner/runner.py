import os
import numpy as np
import json
from utils import NumpyEncoder
import time

class Runner:
    def __init__(self, agent, frequency, env, save_frequency=3000, logging=100, total_episode=1e10, curriculum=None,
                 frequency_mode='episodes', random_actions=None,
                 # IRL
                 reward_model=None, fixed_reward_model=False, dems_name='', reward_frequency=30,
                 **kwargs):

        # Runner objects and parameters
        self.agent = agent
        self.curriculum = curriculum
        self.total_episode = total_episode
        self.frequency = frequency
        self.frequency_mode = frequency_mode
        self.random_actions = random_actions
        self.logging = logging
        self.save_frequency = save_frequency
        self.env = env

        # Recurrent
        self.recurrent = self.agent.recurrent

        # Objects and parameters for IRL
        self.reward_model = reward_model
        self.fixed_reward_model = fixed_reward_model
        self.dems_name = dems_name
        self.reward_frequency = reward_frequency

        # Initialize reward model
        if self.reward_model is not None:
            if not self.fixed_reward_model:
                # Ask for demonstrations
                answer = None
                while answer != 'y' and answer != 'n':
                    answer = input('Do you want to create new demonstrations? [y/n] ')
                if answer == 'y':
                    dems, vals = self.reward_model.create_demonstrations(env=self.env)
                elif answer == 'p':
                    dems, vals = self.reward_model.create_demonstrations(env=self.env, with_policy=True)
                else:
                    print('Loading demonstrations...')
                    dems, vals = self.reward_model.load_demonstrations(self.dems_name)

                print('Demonstrations loaded! We have ' + str(len(dems['obs'])) + " timesteps in these demonstrations")
                print('and ' + str(len(vals['obs'])) + " timesteps in these validations.")

                # Getting initial experience from the environment to do the first training epoch of the reward model
                self.get_experience(env, self.reward_frequency, random=True)
                self.reward_model.train()

        # Global runner statistics
        # total episode
        self.ep = 0
        # total steps
        self.total_step = 0
        # Initialize history
        # History to save model statistics
        self.history = {
            "episode_rewards": [],
            "episode_timesteps": [],
            "mean_entropies": [],
            "std_entropies": [],
            "reward_model_loss": [],
            "env_rewards": []
        }

        # For curriculum training
        self.start_training = 0
        self.current_curriculum_step = 0

        # If a saved model with the model_name already exists, load it (and the history attached to it)
        if os.path.exists('{}/{}.meta'.format('saved', agent.model_name)):
            answer = None
            while answer != 'y' and answer != 'n':
                answer = input("There's already an agent saved with name {}, "
                               "do you want to continue training? [y/n] ".format(agent.model_name))

            if answer == 'y':
                self.history = self.load_model(agent.model_name, agent)
                self.ep = len(self.history['episode_timesteps'])
                self.total_step = np.sum(self.history['episode_timesteps'])

    def run(self):

        # Trainin loop
        # Start training
        start_time = time.time()
        while self.ep <= self.total_episode:
            # Reset the episode
            self.ep += 1
            step = 0

            # Set actual curriculum
            config = self.set_curriculum(self.curriculum, np.sum(self.history['episode_timesteps']))
            if self.start_training == 0:
                print(config)
            self.start_training = 1
            self.env.set_config(config)

            state = self.env.reset()
            done = False
            # Total reward of the episode
            episode_reward = 0
            # Total reward of the environment, in case of IRL it can be different from the actual reward of the agent
            env_episode_reward = 0

            # Save local entropies
            local_entropies = []

            # If recurrent, initialize hidden state
            if self.recurrent:
                internal = (np.zeros([1, self.agent.recurrent_size]), np.zeros([1, self.agent.recurrent_size]))

            # Episode loop
            while True:

                # Evaluation - Execute step
                if not self.recurrent:
                    action, logprob, probs = self.agent.eval([state])
                else:
                    action, logprob, probs, internal_n = self.agent.eval_recurrent([state], internal)

                # Do a first run of random actions if specified
                if self.random_actions is not None and self.total_step < self.random_actions:
                    action = [np.random.randint(self.agent.action_size)]

                action = action[0]
                # Save probabilities for entropy
                local_entropies.append(self.env.entropy(probs[0]))

                # Execute in the environment
                state_n, done, reward, _ = self.env.execute(action)

                # If exists a reward model, use it instead of the env reward
                if self.reward_model is not None:
                    env_episode_reward += reward
                    # If we use a trained reward model, use a simple eval without updating it
                    if self.fixed_reward_model:
                        reward = self.reward_model.eval([state], [state_n], [action])  # , probs=[probs[actions]])
                    # If not, eval with discriminator and update its buffer for training it
                    else:
                        reward = self.reward_model.eval_discriminator([state], [state_n], [probs[0][action]], [action])
                        self.reward_model.add_to_buffer(state, state_n, action)

                # If step is equal than max timesteps, terminate the episode
                if step >= self.env._max_episode_timesteps - 1:
                    done = True

                # Get the cumulative reward
                episode_reward += reward

                # Update PPO memory
                if not self.recurrent:
                    self.agent.add_to_buffer(state, state_n, action, reward, logprob, done)
                else:
                    # If we use a recurrent network, we need to store aso the internal state of the recurrent layer
                    try:
                        self.agent.add_to_buffer(state, state_n, action, reward, logprob, done,
                                                 internal.c[0], internal.h[0])
                    except Exception as e:
                        zero_state = np.reshape(internal[0], [-1,])
                        self.agent.add_to_buffer(state, state_n, action, reward, logprob, done,
                                                 zero_state, zero_state)
                    internal = internal_n
                state = state_n

                step += 1
                self.total_step += 1

                # If frequency timesteps are passed, update the policy
                if self.frequency_mode == 'timesteps' and \
                        self.total_step > 0 and self.total_step % self.frequency == 0:
                    if self.random_actions is not None:
                        if self.total_step > self.random_actions:
                            self.agent.train()
                    else:
                        self.agent.train()

                # If done, end the episode and save statistics
                if done:
                    self.history['episode_rewards'].append(episode_reward)
                    self.history['episode_timesteps'].append(step)
                    self.history['mean_entropies'].append(np.mean(local_entropies))
                    self.history['std_entropies'].append(np.std(local_entropies))
                    self.history['env_rewards'].append(env_episode_reward)
                    break

            # Logging information
            if self.ep > 0 and self.ep % self.logging == 0:
                print('Mean of {} episode reward after {} episodes: {}'.
                      format(self.logging, self.ep, np.mean(self.history['episode_rewards'][-self.logging:])))

                if self.reward_model is not None:
                    print('Mean of {} environment episode reward after {} episodes: {}'.
                            format(self.logging, self.ep, np.mean(self.history['env_rewards'][-self.logging:])))

                print('The agent made a total of {} steps'.format(self.total_step))

                self.timer(start_time, time.time())

            # If frequency episodes are passed, update the policy
            if self.frequency_mode == 'episodes' and self.ep > 0 and self.ep % self.frequency == 0:
                self.agent.train()

            # If IRL, update the reward model after reward_frequency episode
            if self.reward_model is not None:
                if not self.fixed_reward_model and self.ep > 0 and self.ep % self.reward_frequency == 0:
                    self.reward_model.train()

            # Save model and statistics
            if self.ep > 0 and self.ep % self.save_frequency == 0:
                self.save_model(self.history, self.agent.model_name, self.curriculum, self.agent)
                if self.reward_model is not None:
                    if not self.fixed_reward_model:
                        self.reward_model.save_model('{}_{}'.format(self.agent.model_name, self.ep))


    def save_model(self, history, model_name, curriculum, agent):

        # Save statistics as json
        json_str = json.dumps(history, cls=NumpyEncoder)
        f = open("arrays/{}.json".format(model_name), "w")
        f.write(json_str)
        f.close()

        # Save curriculum as json
        json_str = json.dumps(curriculum, cls=NumpyEncoder)
        f = open("arrays/{}_curriculum.json".format(model_name), "w")
        f.write(json_str)
        f.close()

        # Save the tf model
        agent.save_model(name=model_name, folder='saved')
        print('Model saved with name: {}'.format(model_name))

    def load_model(self, model_name, agent):
        agent.load_model(name=model_name, folder='saved')
        with open("arrays/{}.json".format(model_name)) as f:
            history = json.load(f)

        return history

    # Update curriculum for DeepCrawl
    def set_curriculum(self, curriculum, total_timesteps, mode='steps'):
        
        if curriculum == None:
            return None

        if mode == 'steps':
            lessons = np.cumsum(curriculum['thresholds'])

            curriculum_step = 0

            for (index, l) in enumerate(lessons):

                if total_timesteps > l:
                    curriculum_step = index + 1

        parameters = curriculum['parameters']
        config = {}

        for (par, value) in parameters.items():
            config[par] = value[curriculum_step]

        self.current_curriculum_step = curriculum_step

        return config

    # For IRL, get initial experience from environment, the agent act in the env without update itself
    def get_experience(self, env, num_discriminator_exp=None, verbose=False, random=False):

        if num_discriminator_exp == None:
            num_discriminator_exp = self.frequency

        # For policy update number
        for ep in range(num_discriminator_exp):
            states = []
            state = env.reset()
            step = 0
            # While the episode si not finished
            reward = 0
            while True:
                step += 1
                action, _, c_probs = self.agent.eval([state])
                if random:
                    num_actions = env.actions()['num_values']
                    action = np.random.randint(0, num_actions)
                state_n, terminal, step_reward, _ = env.execute(actions=action)
                self.reward_model.add_to_buffer(state, state_n, action)

                state = state_n
                reward += step_reward
                if terminal or step >= env._max_episode_timesteps:
                    break

            if verbose:
                print("Reward at the end of episode " + str(ep + 1) + ": " + str(reward))

    # Method for count time after each episode
    def timer(self, start, end):
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
