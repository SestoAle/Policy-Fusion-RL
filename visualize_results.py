import matplotlib.pyplot as plt
import json
import numpy as np

import argparse
import glob

parser = argparse.ArgumentParser()

parser.add_argument('-mn', '--models-name', help="The name of the model", default='lstm, *no_norm*')
parser.add_argument('-nm', '--num-mean', help="The number of the episode to compute the mean", default=100)
parser.add_argument('-mr', '--num-mean-reward-loss', help="Same as nm, for reward loss", default=10)
parser.add_argument('-sp', '--save-plot', help="If true save the plot in folder saved_plot", default=None)
parser.add_argument('-ep', '--episodes', help="Number of the episodes to observe", default=9000)

parser.add_argument('-xa', '--x-axis', help="Number of the episodes to observe", default='episode')
args = parser.parse_args()


models_name = args.models_name
while models_name == "" or models_name == " " or models_name == None:
    models_name = input('Insert model name: ')

models_name = models_name.replace(' ', '')
models_name = models_name.replace('.json', '')
models_name = models_name.split(",")

histories = []
filenames = []
for model_name in models_name:
    path = glob.glob("arrays/" + model_name + ".json")
    for filename in path:
        if 'curriculum' in filename:
            continue
        with open(filename, 'r') as f:
            histories.append(json.load(f))
            filenames.append(filename)

episodes = args.episodes
if episodes is not None:
    episodes = int(episodes)

print(filenames)
models_name = []
for (index, filename) in enumerate(filenames):
    models_name.append(filename.replace('arrays/', '').replace('.json', ''))

#models_name = ['Learnt RF in ProcEnv', 'AIRL in ProcEnv']

episodes_rewards = []
means_entropies = []
episodes_successes = []
reward_model_losses = []
timestepss = []
rm_episodes_rewards = []


i = 0
for history in histories:
    i += 1

    episodes_reward = np.asarray(history.get("env_rewards", list()))

    rm_episodes_reward = []
    if np.mean(episodes_reward) == 0 or len(episodes_reward) == 0:
       episodes_reward = np.asarray(history.get("episode_rewards", list()))
    else:
        rm_episodes_reward = np.asarray(history.get("episode_rewards", list()))

    tot_episodes = len(episodes_reward)
    episodes_reward = episodes_reward[:episodes]
    waste = np.alen(episodes_reward)%args.num_mean
    waste = -np.alen(episodes_reward) if waste == 0 else waste
    episodes_reward = episodes_reward[:-waste]
    rm_episodes_reward = rm_episodes_reward[:-waste]
    mean_entropies = np.asarray(history.get("mean_entropies", list()))[:episodes][:-waste]
    std_entropies = np.asarray(history.get("std_entropies", list()))[:episodes][:-waste]
    episodes_success = episodes_reward > 0
    episodes_timesteps = np.asarray(history.get("episode_timesteps", list()))[:episodes][:-waste]
    timesteps = np.asarray(history.get("episode_timesteps", list()))[:episodes][:-waste]

    reward_model_loss = np.asarray(history.get("reward_model_loss", list()))
    tot_updates = len(reward_model_loss)
    if tot_updates > 0:
        num_ep_for_update = int(tot_episodes/tot_updates)
        loss_episodes = int(len(episodes_reward)/num_ep_for_update)
        reward_model_loss = reward_model_loss[:loss_episodes]
        waste_reward_model_loss = np.alen(reward_model_loss)%args.num_mean_reward_loss
        waste_reward_model_loss = -np.alen(reward_model_loss) if waste_reward_model_loss == 0 else waste_reward_model_loss
        reward_model_loss = reward_model_loss[:-waste_reward_model_loss]
    cum_timesteps = np.cumsum(timesteps)

    episodes_rewards.append(episodes_reward)
    means_entropies.append(mean_entropies)
    episodes_successes.append(episodes_success)
    reward_model_losses.append(reward_model_loss)
    rm_episodes_rewards.append(rm_episodes_reward)
    timestepss.append(timesteps)

num_mean = int(args.num_mean)
num_mean_reward_loss = int(args.num_mean_reward_loss)
save_plot = bool(args.save_plot)

print("Mean of " + str(num_mean) + " episodes")

model_name = ''
for name in models_name:
    model_name += (name + '_')


plt.figure(1)
plt.title("Reward")
nums_episodes = []
for episodes_reward, model_name, timesteps in zip(episodes_rewards, models_name, timestepss):
    num_episodes = np.asarray(
        range(1, np.size(np.mean(episodes_reward.reshape(-1, num_mean), axis=1)) + 1)) * num_mean

    nums_episodes.append(num_episodes)

    if args.x_axis == 'timesteps':
        x = np.mean(np.cumsum(timesteps).reshape(-1, num_mean), axis=1)
    else:
        x = num_episodes
    plt.plot(x, np.mean(episodes_reward.reshape(-1, num_mean), axis=1), linestyle='solid')

plt.legend(models_name)
if args.x_axis == 'timesteps':
    plt.xlabel("Timesteps")
else:
    plt.xlabel("Episodes")
plt.ylabel("Mean Reward")
if save_plot:
    plt.savefig("saved_plots/" + model_name + "_reward.png", dpi=300)

# Mean of RL policy
#plt.hlines(13.5, 0, max([len(ne) for ne in episodes_rewards]), linestyles='dashed', label='Expert Policy')
#plt.hlines(3.8, 0, max([len(ne) for ne in episodes_rewards]), linestyles='dashed', label='Expert Policy')
#plt.hlines(145, 0, max([len(ne) for ne in episodes_rewards]), linestyles='dashed', label='Expert Policy')
#plt.hlines(2.8, 0, max([len(ne) for ne in episodes_rewards]), linestyles='dashed', label='Expert Policy')

plt.figure(2)
plt.title("Entropy")
for (mean_entropies, num_episodes, timesteps) in zip(means_entropies, nums_episodes, timestepss):
    if args.x_axis == 'timesteps':
        x = np.mean(np.cumsum(timesteps).reshape(-1, num_mean), axis=1)
    else:
        x = num_episodes
    plt.plot(x, np.mean(mean_entropies.reshape(-1, num_mean), axis=1))
plt.legend(models_name)
if args.x_axis == 'timesteps':
    plt.xlabel("Timesteps")
else:
    plt.xlabel("Episodes")
plt.ylabel("Mean Entropy")
if save_plot:
    plt.savefig("saved_plots/" + model_name + "_entropy.png", dpi=300)

plt.figure(3)
plt.title("Success")
for (episodes_success, num_episodes, timesteps) in zip(episodes_successes, nums_episodes, timestepss):
    if args.x_axis == 'timesteps':
        x = np.mean(np.cumsum(timesteps).reshape(-1, num_mean), axis=1)
    else:
        x = num_episodes
    plt.plot(x, np.mean(episodes_success.reshape(-1, num_mean), axis=1))
plt.legend(models_name)
if args.x_axis == 'timesteps':
    plt.xlabel("Timesteps")
else:
    plt.xlabel("Episodes")

plt.ylabel("Success Rate")
if save_plot:
    plt.savefig("saved_plots/" + model_name + "_success.png", dpi=300)

legends = []
for (reward_model_loss, num_episodes, timesteps, episodes_reward, model_name) in zip(reward_model_losses, nums_episodes, timestepss, episodes_rewards, models_name):
    if len(reward_model_loss) > 0:
        plt.figure(4)
        plt.title("Reward Loss")

        reward_model_loss = np.mean(reward_model_loss.reshape(-1, num_mean_reward_loss), axis=1)

        if args.x_axis == 'timesteps':
            x = np.mean(np.cumsum(timesteps).reshape(np.shape(reward_model_loss)[0], -1), axis=1)
        else:
            num_reward_updates = np.asarray(range(len(reward_model_loss)))
            num_reward_updates = num_reward_updates * int(len(episodes_reward) / len(reward_model_loss))
            x = num_reward_updates

        legends.append(model_name)
        plt.plot(x, reward_model_loss)
        if args.x_axis == 'timesteps':
            plt.xlabel("Timesteps")
        else:
            plt.xlabel("Episodes")
        plt.ylabel("Loss")
        if save_plot:
            plt.savefig("saved_plots/" + model_name + "_reward_model_loss.png", dpi=300)

if len(legends) > 0:
    plt.legend(legends)

legends = []
for (rm_episodes_reward, num_episodes, timesteps, model_name) in zip(rm_episodes_rewards, nums_episodes, timestepss, models_name):
    if len(rm_episodes_reward) > 0:
        plt.figure(5)
        plt.title("RM Reward")
        if args.x_axis == 'timesteps':
            x = np.mean(np.cumsum(timesteps).reshape(-1, num_mean), axis=1)
        else:
            x = num_episodes
        plt.plot(x, np.mean(rm_episodes_reward.reshape(-1, num_mean), axis=1))
        legends.append(model_name)
        if args.x_axis == 'timesteps':
            plt.xlabel("Timesteps")
        else:
            plt.xlabel("Episodes")
        plt.ylabel("Mean Entropy")

if len(legends) > 0:
    #plt.legend(legends)

    plt.legend(legends, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


for timesteps, episodes_reward, model_name in zip(timestepss, episodes_rewards, models_name):
    print(model_name + ' max reward: ' + str(np.max(np.mean(episodes_reward.reshape(-1, num_mean), axis=1))))
    print(model_name + ' min reward: ' + str(np.min(np.mean(episodes_reward.reshape(-1, num_mean), axis=1))))
    print("Number of timesteps: " + str(np.sum(timesteps)))
    print("Number of episodes: " + str(np.size(episodes_reward)))

    print('Mean of the last 100 episodes: ' + str(np.mean(episodes_reward[-100:])))

plt.show()
