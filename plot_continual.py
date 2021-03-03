import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
import re
import os

sns.set_theme(style="dark")
sns.set(font="Times New Roman", font_scale=1.5)

eps = 1e-12

import argparse
import glob

# Simple script to replicate the plot in the paper
# The script will check the file inside reward_experiments folder

parser = argparse.ArgumentParser()

parser.add_argument('-mn', '--models-name', help="The name of the model", default='*mini*')

args = parser.parse_args()

plots = args.models_name.split(";")

legends = []
f, (x1, x2) = plt.subplots(2,1, figsize=(10,6))
f.tight_layout(pad=0.5)
for (i,plot) in enumerate(plots):

    plot_title = ''

    rewards = []
    filenames = []

    models_name = plot
    models_name = models_name.replace(' ', '')
    models_name = models_name.replace('.json', '')
    models_name = models_name.split(",")

    for model_name in models_name:
        path = glob.glob("reward_experiments/" + model_name + ".json")
        path.sort()
        for filename in path:
            with open(filename, 'r') as f:
                filenames.append(filename)
                rewards.append(json.load(f))

    keys = rewards[0].keys()

    percentages = []
    all_data = []
    i = 0
    for k in keys:
        if k == 'reward_2':
            continue
        all_rews = []
        data = []
        for r_dict in rewards:
            length = 0
            episode_rewards = []

            for r in r_dict[k]:
                length += len(r)
                current_rews = r
                episode_rewards.append(np.sum(current_rews))
            data.append(np.mean(episode_rewards))
            all_rews.extend(episode_rewards)

        data = np.array(data)
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        all_data.append(data)
        x1.plot(range(len(data)), data, '-o', ms=12, linewidth=4)

        legends.append("$R_{}$".format(i))
        i += 1

    x1.set_xticks([])
    x1.legend(legends)


try:
    # Total reward
    percentages = []

    for r in rewards:
        tmp_r = []
        for v in r['reward_2']:
            tmp_r.append(np.sum(v))
        percentages.append(np.mean(tmp_r))

    pal = sns.color_palette("Reds_d", len(percentages))
    rank = data.argsort().argsort()
    x = np.array(range(len(percentages)))
    x2.legend('win rate')
    sns.barplot(x=x, y=(np.array(percentages)), palette=np.array(pal[::-1])[rank], ax=x2)
    for p in x2.patches:
        x2.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center',
                       va = 'center', xytext = (0, 10), textcoords = 'offset points')
    plt.setp(x2.patches, linewidth=1.5, edgecolor='black')
except Exception as e:
    pass

labels = []
for filename in filenames:
    if '_ew' in filename:
        labels.append('EW')
    if '_et' in filename:
        labels.append('ET')
    if '_mp' in filename:
        labels.append('MP')
    if '_pp' in filename:
        labels.append('PP')
    if '_main' in filename:
        labels.append('Main\nPolicy')
    if '_ft' in filename:
        labels.append('Fine\nTuning')
    if '_scratch' in filename:
        labels.append('From\nScratch')

x2.set_xticklabels(labels)

x1.set_title('Normalized Rewards', pad=20)
x2.set_title('Total Reward', pad=20)
sns.despine()
plt.show()