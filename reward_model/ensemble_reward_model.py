import numpy as np
import pickle
#from reward_model.utils import LimitedRunningStat, RunningStat
from deepcrawl.reward_model.reward_model import RewardModel

class EnsembleRewardModel:

    def __init__(self, actions_size, policy, sess = None, gamma=0.9, lr=1e-5, use_vairl=False,
                 mutual_information=0.5, alpha=0.0005, with_global=True, with_action=False, name='reward_model',
                 entropy_weight = 0.1, with_value=True, vs=None, num_models = 5, **kwargs):


        self.num_models = num_models
        self.models = [RewardModel(actions_size, policy, sess, gamma, lr, use_vairl,
                 mutual_information, alpha, with_global, with_action, name + '_' + str(i),
                 entropy_weight, with_value, vs) for i in range(num_models)]



    def fit(self, expert_traj, policy_traj, num_itr=20, batch_size=32):

        m_losses = []
        for model in self.models:

            loss, _ = model.fit(expert_traj, policy_traj, num_itr, batch_size)
            m_losses.append(loss)

        return np.mean(m_losses), 0

    def eval(self, obs, obs_n, acts=None):

        m_rewards = []
        for model in self.models:
            reward = model.eval(obs, obs_n, acts)
            m_rewards.append(reward)

        return m_rewards

    def eval_discriminator(self, obs, obs_n, probs, acts=None, norm=True):

        m_fs = []
        m_values = []
        m_values_n = []

        for model in self.models:
            f, disc, value, value_n, r_feat, v_feat, vn_feat = model.eval_discriminator(obs, obs_n, probs, acts)
            m_fs.append(f)
            m_values.append(value)
            m_values_n.append(value_n)

        return m_fs, m_values, m_values_n

    # Load demonstrations from file
    def load_demonstrations(self, name='dems.pkl'):
        with open('reward_model/dems/' + name, 'rb') as f:
            expert_traj = pickle.load(f)

        with open('reward_model/dems/vals.pkl', 'rb') as f:
            val_traj = pickle.load(f)

        self.set_demonstrations(expert_traj, val_traj)

        return expert_traj, val_traj

    # Update demonstrations
    def set_demonstrations(self, demonstrations, validations):

        self.expert_traj = demonstrations

        if validations is not None:
            self.validation_traj = validations

    # Save the entire model
    def save_model(self, name=None):
        for (i,model) in enumerate(self.models):
            model.save_model('ensemble_' + name + '_' + str(i))

    # Load entire model
    def load_model(self, name=None):

        for (i,model) in enumerate(self.models):
            model.load_model('ensemble_' + name + '_' + str(i))

    # Create and return some demonstrations [(states, actions, frames)]. The order of the demonstrations must be from
    # best to worst. The number of demonstrations is given by the user
    def create_demonstrations(self, env, save_demonstrations=True, inference=False, verbose=False,
                              with_policy=False, num_episode=11, max_timestep=20,
                              save_rew_in_sampled_env=False, sampled_envs=20, model_name='model', random=False
                              ):
        end = False

        # Initialize trajectories buffer
        expert_traj = {
            'obs': [],
            'obs_n': [],
            'acts': [],
        }

        val_traj = {
            'obs': [],
            'obs_n': [],
            'acts': []
        }

        if with_policy is None:
            num_episode = None

        episode = 1

        rewards = dict()
        gt_rewards = dict()
        disc_probs = dict()
        policy_probs = dict()
        values = dict()
        values_n = dict()

        r_feats = dict()
        v_feats = dict()
        vn_feats = dict()

        while not end:
            # Make another demonstration
            print('Demonstration n° ' + str(episode))
            # Reset the environment
            state = env.reset()
            states = [state]
            actions = []
            done = False
            step = 0
            cum_reward = 0
            # New sequence of states and actions
            while not done:
                #try:
                    # Input the action and save the new state and action
                    step += 1
                    if not save_rew_in_sampled_env:
                        print("Timestep: " + str(step))
                    if verbose:
                        env.print_observation(state)
                    if not with_policy:
                        action = input('action: ')
                        if action == "f":
                            done = True
                            continue
                        while env.command_to_action(action) >= 99:
                            action = input('action: ')
                    else:
                        if random:
                            action = np.random.randint(0, 8)
                        else:
                            action, probs = self.models[0].select_action(state)

                    state_n, done, reward = env.execute(action)

                    cum_reward += reward
                    real_rew = reward
                    if not with_policy:
                        action = env.command_to_action(action)
                    # If inference is true, print the reward
                    if inference:
                        _, probs = self.models[0].select_action(state)
                        reward = self.eval([state], [state_n], [action])
                        _, value, value_n = self.eval_discriminator([state], [state_n], [probs[action]], [action])
                        if save_rew_in_sampled_env:

                            initial_state = (episode - 1) % sampled_envs
                            if initial_state in rewards:
                                gt_rewards[initial_state].extend([real_rew])
                                rewards[initial_state].extend([reward])
                                policy_probs[initial_state].extend([probs])
                                values[initial_state].extend([value])
                                values_n[initial_state].extend([value_n])
                            else:
                                gt_rewards[initial_state] = [real_rew]
                                rewards[initial_state] = [reward]
                                policy_probs[initial_state] = [probs]
                                values[initial_state] = [value]
                                values_n[initial_state] = [value_n]


                        if not save_rew_in_sampled_env:
                            # print('Discriminator probability: ' + str(disc))
                            print('Unnormalize reward: ' + str(reward))
                            # reward = self.normalize_rewards(reward)
                            print('Normalize reward: ' + str(reward))
                            print('Probability of state space: ')
                            print(probs)
                    state = state_n
                    states.append(state)
                    actions.append(action)
                    if step >= max_timestep:
                        done = True
                #except Exception as e:
                #    print(e)
                #    continue

            if not inference:
                y = None
                if with_policy:
                    if episode < num_episode:
                        if step < 80:
                            y = 'y'
                        else:
                            y = 'n'
                while y != 'y' and y != 'n':
                    y = input('Do you want to save this demonstration? [y/n] ')

                if y == 'y':
                    # Update expert trajectories
                    expert_traj['obs'].extend(np.array(states[:-1]))
                    expert_traj['obs_n'].extend(np.array(states[1:]))
                    expert_traj['acts'].extend(np.array(actions))
                    episode += 1
                else:
                    if step < 80:
                        y = input('Do you want to save this demonstration as validation? [y/n] ')
                    else:
                        y = 'n'
                    if y == 'y':
                        val_traj['obs'].extend(np.array(states[:-1]))
                        val_traj['obs_n'].extend(np.array(states[1:]))
                        val_traj['acts'].extend(np.array(actions))
                        episode += 1

            if save_rew_in_sampled_env:
                episode += 1

            y = None
            if num_episode is None:
                while y != 'y' and y != 'n':
                    if not inference:
                        y = input('Do you want to create another demonstration? [y/n] ')
                    else:
                        y = input('Do you want to try another episode? [y/n] ')

                    if y == 'n':
                        end = True
            else:
                if episode >= num_episode + 1:
                    end = True

        if len(val_traj['obs']) <= 0:
            val_traj = None

        # Save demonstrations to file
        if save_demonstrations and not inference:
            print('Saving the demonstrations...')
            self.save_demonstrations(expert_traj, val_traj)
            print('Demonstrations saved!')

        if not inference:
            self.set_demonstrations(expert_traj, val_traj)

        if save_rew_in_sampled_env:
            with open('reward_model/rewards/' + model_name + '_rews.pkl', 'wb') as f:
                pickle.dump(rewards, f, pickle.HIGHEST_PROTOCOL)
            with open('reward_model/rewards/' + model_name + '_disc_probs.pkl', 'wb') as f:
                pickle.dump(disc_probs, f, pickle.HIGHEST_PROTOCOL)
            with open('reward_model/rewards/' + model_name + '_policy_probs.pkl', 'wb') as f:
                pickle.dump(policy_probs, f, pickle.HIGHEST_PROTOCOL)
            with open('reward_model/rewards/' + model_name + '_values.pkl', 'wb') as f:
                pickle.dump(values, f, pickle.HIGHEST_PROTOCOL)
            with open('reward_model/rewards/' + model_name + '_values_n.pkl', 'wb') as f:
                pickle.dump(values_n, f, pickle.HIGHEST_PROTOCOL)
            with open('reward_model/rewards/' + model_name + '_gt_rews.pkl', 'wb') as f:
                pickle.dump(gt_rewards, f, pickle.HIGHEST_PROTOCOL)
            with open('reward_model/rewards/' + model_name + '_r_feats.pkl', 'wb') as f:
                pickle.dump(r_feats, f, pickle.HIGHEST_PROTOCOL)
            with open('reward_model/rewards/' + model_name + '_v_feats.pkl', 'wb') as f:
                pickle.dump(v_feats, f, pickle.HIGHEST_PROTOCOL)
            with open('reward_model/rewards/' + model_name + '_vn_feats.pkl', 'wb') as f:
                pickle.dump(vn_feats, f, pickle.HIGHEST_PROTOCOL)

        return expert_traj, val_traj
