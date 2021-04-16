from agents.PPO import PPO
from runner.runner import Runner
import os
import tensorflow_probability as tfp
import time
import tensorflow as tf
import argparse
from miniworld_env_wrapper import Miniworld
import numpy as np
import json
import re
from utils import NumpyEncoder

from reward_model.reward_model import RewardModel


eps = 1e-5

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Parse arguments for training
parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model-name', help="The name of the model", default='mini_box')
parser.add_argument('-mt', '--max-timesteps', help="Max timestep per episode", default=400)

# Test reward models
parser.add_argument('-fm', '--fusion-mode', help="IRL", choices=['mp', 'pp', 'et', 'ew'], default="ew")
parser.add_argument('-t', '--temperatures', help="IRL", default="1.0,1.0,1.0")
parser.add_argument('-sn', '--save-name', help="The name for save the results", default=None)


args = parser.parse_args()

def boltzmann(probs, temperature = 1.):
    sum = np.sum(np.power(probs, 1/temperature))
    new_probs = []
    for p in probs:
        new_probs.append(np.power(p, 1/temperature) / sum)

    return np.asarray(new_probs)

def entropy(probs):
    entr = np.sum(probs * np.log(probs + 1e-12))
    return -entr

def rewardToText(key):
    if key == 'reward_0':
        return 'reward box'
    if key == 'reward_1':
        return 'reward ball'
    if key == 'reward_2':
        return 'complete reward'


if __name__ == "__main__":

    model_name = args.model_name
    max_episode_timestep = int(args.max_timesteps)

    # Curriculum structure; here you can specify also the agent statistics (ATK, DES, DEF and HP)
    curriculum = None

    # Total episode of training
    total_episode = 100

    # Open the environment with all the desired flags
    env = Miniworld(reward_type='box', )
    # Load the agents
    agents = []
    models = args.model_name.split(",")
    for i,m in enumerate(models):
        # Create agent
        graph = tf.compat.v1.Graph()
        with graph.as_default():
            tf.compat.v1.disable_eager_execution()
            sess = tf.compat.v1.Session(graph=graph)
            agent = PPO(sess=sess, model_name=model_name, recurrent=False)
            # Load agent
            agent.load_model(m, 'saved')
            agents.append(agent)

    if len(models) == 1:
        args.fusion_mode = 'mp'

    # Create the reward models
    reward_models = []
    sessions = []

    try:
        # Evaluation loop
        current_episode = 0
        num_reward_models = len(reward_models)
        episode_rewards = dict()
        step_rewards = dict()
        all_step_rewards = dict()

        entropies = []
        rang = 0
        for i in range(3):
            episode_rewards["reward_{}".format(i)] = []
            all_step_rewards["reward_{}".format(i)] = []

        while current_episode < total_episode:
            done = False
            current_step = 0
            for i in range(3):
                step_rewards["reward_{}".format(i)] = []
            state = env.reset()
            action = 0
            while not done:

                if args.fusion_mode == 'pp':
                    total_probs = np.ones(5)
                elif args.fusion_mode == 'mp':
                    total_probs = np.zeros(5)

                main_entropy = np.inf
                min_entropy = np.inf
                min_entropy_idx = np.inf
                temperatures = [float(t) for t in args.temperatures.split(",")]
                for (i, agent) in enumerate(agents):
                    _, _, probs = agent.eval([state])
                    probs = probs[0]
                    if args.fusion_mode == 'pp':

                        probs = boltzmann(probs, temperatures[i])
                        total_probs *= probs
                    elif args.fusion_mode == 'mp':
                        probs = boltzmann(probs, temperatures[i])
                        total_probs += probs
                    elif args.fusion_mode == 'et':
                        if i == 0:
                            main_entropy = entropy(probs)
                            continue
                        if entropy(probs) < min_entropy:
                            min_entropy = entropy(probs)
                            min_entropy_idx = i
                    elif args.fusion_mode == 'ew':
                        if i == 0:
                            main_entropy = entropy(probs)
                            continue
                        if entropy(probs) < min_entropy:
                            min_entropy = entropy(probs)
                            min_entropy_idx = i
                if args.fusion_mode == 'mp':
                    total_probs /= (num_reward_models + 1)

                if args.fusion_mode == 'et':
                    if min_entropy < main_entropy + 5.0:
                        total_probs = boltzmann(agents[min_entropy_idx].eval([state])[2][0], 1.0)
                        action = np.argmax(np.random.multinomial(1, total_probs))
                    else:
                        total_probs = boltzmann(agents[0].eval([state])[2][0], 1.0)
                        action = np.argmax(np.random.multinomial(1, total_probs))
                elif args.fusion_mode == 'ew':
                    entropies.append(min_entropy)
                    
                    min_entropy = (min_entropy - 0.0) / (1.61 - 0.0)
                    min_entropy = np.clip(min_entropy, 0, 1)
                    main_probs = agents[0].eval([state])[2] * min_entropy
                    sub_probs = boltzmann(agents[min_entropy_idx].eval([state])[2], 5.0)
                    main_probs += (sub_probs * (1. - min_entropy))
                    main_probs = boltzmann(main_probs[0], 1.0)
                    action = np.argmax(np.random.multinomial(1, main_probs))
                else:
                    total_probs = boltzmann(total_probs, 1.0)
                    action = np.argmax(np.random.multinomial(1, total_probs))

                state_n, done, reward, info = env.execute(action)
                state = state_n

                # Saving rewards. 0: reward_box, 1: reward_ball, 2:reward_all
                for (i, rew) in enumerate(info):
                    r = info[rew]
                    step_rewards["reward_{}".format(i)].append(r)

                current_step += 1
                if current_step >= max_episode_timestep:
                    done = True

            for i in range(len(step_rewards.keys())):
                all_step_rewards["reward_{}".format(i)].append(step_rewards["reward_{}".format(i)])

            current_episode += 1
            print("Episode {} finished".format(current_episode))

        print("End of testing phase: ")

        for key in all_step_rewards.keys():
            episode_rewards = []
            for r in all_step_rewards[key]:
                episode_rewards.append(np.sum(r))

            print('mean of {}: {}'.format(rewardToText(key), np.mean(episode_rewards)))

        if args.save_name is not None:
            print("Saving the experiment..")
            json_str = json.dumps(all_step_rewards, cls=NumpyEncoder)
            f = open('reward_experiments/{}_{}.json'.format(args.save_name,args.fusion_mode), "w")
            f.write(json_str)
            f.close()
            print("Experiment saved with name {}!".format(args.save_name))

    finally:
        env.close()

