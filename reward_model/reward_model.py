import tensorflow as tf
import numpy as np
import pickle
#from reward_model.utils import LimitedRunningStat, RunningStat
from utils import DynamicRunningStat, LimitedRunningStat, RunningStat
import random
from math import sqrt
from utils import *

eps = 1e-12

class RewardModel:

    def __init__(self, actions_size, policy, sess = None, gamma=0.99, lr=1e-5, batch_size=32, num_itr=20,
                 use_vairl=False, mutual_information=0.5, alpha=0.0005, with_action=False, name='reward_model',
                 entropy_weight = 0.5, with_value=True, fixed_reward_model=False,
                 vs=None, **kwargs):

        # Initialize some model attributes
        # RunningStat to normalize reward from the model
        if not fixed_reward_model:
            self.r_norm = DynamicRunningStat()
        else:
            self.r_norm = RunningStat(1)

        # Discount factor
        self.gamma = gamma
        # Policy agent needed to compute the discriminator
        self.policy = policy
        # Demonstrations buffer
        self.expert_traj = None
        self.validation_traj = None
        # Num of actions available in the environment
        self.actions_size = actions_size
        # If is state-only or state-action discriminator
        self.with_action = with_action


        # TF parameters
        self.sess = sess
        self.lr = lr
        self.batch_size = batch_size
        self.num_itr = num_itr
        self.entropy_weight = entropy_weight

        # Use Variation Bottleneck Autoencoder
        self.use_vairl = use_vairl
        self.mutual_information = mutual_information
        self.alpha = alpha
        self.name = name

        # Buffer of policy experience with which train the reward model
        self.buffer = dict()
        self.create_buffer()

        with tf.compat.v1.variable_scope(name) as vs:
            with tf.compat.v1.variable_scope('irl'):

                # Input spec for both reward and value function

                # Current state (DeepCrawl spec)
                self.global_state = tf.compat.v1.placeholder(tf.float32, [None, 10, 10, 52], name='global_state')
                self.local_state = tf.compat.v1.placeholder(tf.float32, [None, 5, 5, 52], name='local_state')
                self.local_two_state = tf.compat.v1.placeholder(tf.float32, [None, 3, 3, 52], name='local_two_state')
                self.agent_stats = tf.compat.v1.placeholder(tf.int32, [None, 16], name='agent_stats')
                self.target_stats = tf.compat.v1.placeholder(tf.int32, [None, 15], name='target_stats')
                if self.with_action:
                    self.acts = tf.compat.v1.placeholder(tf.int32, [None, 1], name='acts')

                # Next state (DeepCrawl spec) - for discriminator
                self.global_state_n = tf.compat.v1.placeholder(tf.float32, [None, 10, 10, 52], name='global_state_n')
                self.local_state_n = tf.compat.v1.placeholder(tf.float32, [None, 5, 5, 52], name='local_state_n')
                self.local_two_state_n = tf.compat.v1.placeholder(tf.float32, [None, 3, 3, 52], name='local_two_state_n')
                self.agent_stats_n = tf.compat.v1.placeholder(tf.int32, [None, 16], name='agent_stats_n')
                self.target_stats_n = tf.compat.v1.placeholder(tf.int32, [None, 15], name='target_stats_n')

                # Probability distribution and labels - whether or not this state belongs to expert buffer
                self.probs = tf.compat.v1.placeholder(tf.float32, [None, 1], name='probs')
                self.labels = tf.compat.v1.placeholder(tf.float32, [None, 1], name='labels')

                # For V-AIRL
                self.use_noise = tf.compat.v1.placeholder(
                    shape=[1], dtype=tf.float32, name="noise"
                )

                self.z_sigma_g = None
                self.z_sigma_h = None
                if self.use_vairl:
                    self.z_sigma_g = tf.compat.v1.get_variable(
                        'z_sigma_g',
                        100,
                        dtype=tf.float32,
                        initializer=tf.compat.v1.ones_initializer(),
                    )
                    self.z_sigma_g_sq = self.z_sigma_g * self.z_sigma_g
                    self.z_log_sigma_g_sq = tf.compat.v1.log(self.z_sigma_g_sq + eps)

                    self.z_sigma_h = tf.compat.v1.get_variable(
                        "z_sigma_h",
                        100,
                        dtype=tf.float32,
                        initializer=tf.compat.v1.ones_initializer(),
                    )
                    self.z_sigma_h_sq = self.z_sigma_h * self.z_sigma_h
                    self.z_log_sigma_h_sq = tf.compat.v1.log(self.z_sigma_h_sq + eps)


                # Reward Funvtion
                with tf.compat.v1.variable_scope('reward'):
                    self.reward, self.z_g = self.conv_net(self.global_state, self.local_state, self.local_two_state,
                                                          self.agent_stats, self.target_stats, with_action = self.with_action,
                                                          z_sigma=self.z_sigma_g, use_noise=self.use_noise)

                # Value Function
                if with_value:
                    with tf.compat.v1.variable_scope('value'):
                        self.value, self.z_h = self.conv_net(self.global_state, self.local_state, self.local_two_state,
                                                             self.agent_stats, self.target_stats,
                                                             z_sigma=self.z_sigma_h, use_noise=self.use_noise, with_action=False)
                    with tf.compat.v1.variable_scope('value', reuse=True):
                        self.value_n, self.z_1_h = self.conv_net(self.global_state_n, self.local_state_n,
                                                                 self.local_two_state_n,
                                                                 self.agent_stats_n, self.target_stats_n,
                                                                 z_sigma=self.z_sigma_h, use_noise=self.use_noise, with_action=False)
    
                    self.f = self.reward + self.gamma * self.value_n - self.value
                else:
                    self.f = self.reward

                # Discriminator
                self.discriminator = tf.math.divide(tf.math.exp(self.f), tf.math.add(tf.math.exp(self.f), self.probs))

                # Loss Function
                self.loss = -tf.reduce_mean((self.labels * tf.math.log(self.discriminator + eps)) + (
                            (1 - self.labels) * tf.math.log(1 - self.discriminator + eps)))

                # Loss function modification for V-AIRL
                if self.use_vairl:
                    # Define beta
                    self.beta = tf.compat.v1.get_variable(
                        "airl_beta",
                        [],
                        trainable=False,
                        dtype=tf.float32,
                        initializer=tf.compat.v1.ones_initializer(),
                    )

                    # Number of batch element
                    self.batch = tf.compat.v1.shape(self.z_g)[0]
                    self.batch_index = tf.dtypes.cast(self.batch / 2, tf.int32)

                    self.kl_loss = tf.reduce_mean(
                        -tf.reduce_sum(
                            1
                            + self.z_log_sigma_g_sq
                            - 0.5 * tf.square(
                                self.z_g[0:self.batch_index, :] * self.z_h[0:self.batch_index, :] * self.z_1_h[
                                                                                                    0:self.batch_index,
                                                                                                    :])
                            - 0.5 * tf.square(
                                self.z_g[self.batch_index:, :] * self.z_h[self.batch_index:, :] * self.z_1_h[
                                                                                                  self.batch_index:, :])
                            - tf.exp(self.z_log_sigma_g_sq),
                            1,
                        )
                    )

                    self.loss = self.beta * (self.kl_loss - self.mutual_information) + self.loss

                # Adam optimizer with gradient clipping
                optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
                gradients, variables = zip(*optimizer.compute_gradients(self.loss))
                gradients, _ = tf.compat.v1.clip_by_global_norm(gradients, 1.0)
                self.step = optimizer.apply_gradients(zip(gradients, variables))
                #self.step = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

                if self.use_vairl:
                    self.make_beta_update()

        self.vs = vs
        self.saver = tf.compat.v1.train.Saver(max_to_keep=None)

    ## Layers
    def linear(self, inp, inner_size, name='linear', bias=True, activation = None, init = None):
        with tf.compat.v1.variable_scope(name):

            lin = tf.compat.v1.layers.dense(inp, inner_size, name=name, activation=activation, use_bias=bias,
                                            kernel_initializer=init)
            return lin

    def conv_layer_2d(self, input, filters, kernel_size, strides=(1, 1), padding="SAME", name='conv', activation=None,
                      bias = True):
        with tf.compat.v1.variable_scope(name):

            conv = tf.compat.v1.layers.conv2d(input, filters, kernel_size, strides, padding=padding, name=name,
                                              activation=activation, use_bias=bias)
            return conv

    def embedding(self, input, indices, size, name='embs'):
        with tf.compat.v1.variable_scope(name):
            shape = (indices, size)
            stddev = min(0.1, sqrt(2.0 / (product(xs=shape[:-1]) + shape[-1])))
            initializer = tf.random.normal(shape=shape, stddev=stddev, dtype=tf.float32)
            W = tf.Variable(
                initial_value=initializer, trainable=True, validate_shape=True, name='W',
                dtype=tf.float32, shape=shape
            )
            return tf.nn.tanh(tf.compat.v1.nn.embedding_lookup(params=W, ids=input, max_norm=None))

    # Netowrk specification
    def conv_net(self, global_state, local_state, local_two_state, agent_stats, target_stats, z_sigma=None,
                 use_noise=None, with_action=False):
        

        conv_10 = self.conv_layer_2d(global_state, 32, [1, 1], name='conv_10', activation=tf.nn.tanh)
        conv_11 = self.conv_layer_2d(conv_10, 32, [3, 3], name='conv_11', activation=tf.nn.leaky_relu)
        conv_12 = self.conv_layer_2d(conv_11, 32, [3, 3], name='conv_12', activation=tf.nn.leaky_relu)
        fc11 = tf.reshape(conv_12, [-1,10*10*32])

        embs_41 = tf.nn.tanh(self.embedding(agent_stats, 129, 32, name='embs_41'))
        embs_41 = tf.reshape(embs_41, [-1, 16 * 32])
        fc_41 = self.linear(embs_41, 100, name = 'fc_41', activation=tf.nn.leaky_relu)

        embs_51 = self.embedding(target_stats, 125, 32, name='embs_51')
        embs_51 = tf.reshape(embs_51, [-1, 15 * 32])
        fc_51 = self.linear(embs_51, 100, name = 'fc_51', activation = tf.nn.leaky_relu)

        all_flat = tf.concat([fc11, fc_41, fc_51], axis=1)

        all_flat = self.linear(all_flat, 32, name='fc1', activation=tf.nn.leaky_relu)

        if with_action:
            hot_acts = tf.one_hot(self.acts, self.actions_size)
            hot_acts = tf.reshape(hot_acts, [-1, self.actions_size])
            all_flat = tf.concat([all_flat, hot_acts], axis=1)

        z_mean = None
        fc2 = self.linear(all_flat, 32, name='fc2', activation = tf.nn.leaky_relu)

        # In case we want to use V-AIRL
        if self.use_vairl:

            z_mean = self.linear(fc2, 32, name='z_mean', init=tf.compat.v1.initializers.variance_scaling(0.01))
            noise = tf.compat.v1.random_normal(tf.compat.v1.shape(z_mean), dtype=tf.float32)

            z = z_mean + z_sigma * noise * use_noise
            fc2 = z

            return self.linear(fc2, 1, name='out'), z_mean
        else:
            return self.linear(fc2, 1, name='out'), None

    # Train method of the discriminator
    def train(self):

        losses = []

        # Update discriminator
        for it in range(self.num_itr):

            expert_batch_idxs = random.sample(range(len(self.expert_traj['obs'])), self.batch_size)
            policy_batch_idxs = random.sample(range(len(self.buffer['obs'])), self.batch_size)

            #expert_batch_idxs = np.random.randint(0, len(expert_traj['obs']), batch_size)
            #policy_batch_idxs = np.random.randint(0, len(policy_traj['obs']), batch_size)

            expert_obs = [self.expert_traj['obs'][id] for id in expert_batch_idxs]
            policy_obs = [self.buffer['obs'][id] for id in policy_batch_idxs]

            expert_obs_n = [self.expert_traj['obs_n'][id] for id in expert_batch_idxs]
            policy_obs_n = [self.buffer['obs_n'][id] for id in policy_batch_idxs]

            expert_acts = [self.expert_traj['acts'][id] for id in expert_batch_idxs]
            policy_acts = [self.buffer['acts'][id] for id in policy_batch_idxs]

            expert_probs = []
            for (index, state) in enumerate(expert_obs):
                _, probs = self.select_action(state)
                expert_probs.append(probs[expert_acts[index]])

            policy_probs = []
            for (index, state) in enumerate(policy_obs):
                _, probs = self.select_action(state)
                policy_probs.append(probs[policy_acts[index]])

            expert_probs = np.asarray(expert_probs)
            policy_probs = np.asarray(policy_probs)

            labels = np.ones((self.batch_size, 1))
            labels = np.concatenate([labels, np.zeros((self.batch_size, 1))])

            e_states = self.obs_to_state(expert_obs)
            p_states = self.obs_to_state(policy_obs)

            all_global = np.concatenate([e_states[0], p_states[0]], axis=0)
            all_local = np.concatenate([e_states[1], p_states[1]], axis=0)
            all_local_two = np.concatenate([e_states[2], p_states[2]], axis=0)
            all_agent_stats = np.concatenate([e_states[3], p_states[3]], axis=0)
            all_target_stats = np.concatenate([e_states[4], p_states[4]], axis=0)

            e_states_n = self.obs_to_state(expert_obs_n)
            p_states_n = self.obs_to_state(policy_obs_n)

            all_global_n = np.concatenate([e_states_n[0], p_states_n[0]], axis=0)
            all_local_n = np.concatenate([e_states_n[1], p_states_n[1]], axis=0)
            all_local_two_n = np.concatenate([e_states_n[2], p_states_n[2]], axis=0)
            all_agent_stats_n = np.concatenate([e_states_n[3], p_states_n[3]], axis=0)
            all_target_stats_n = np.concatenate([e_states_n[4], p_states_n[4]], axis=0)

            all_probs = np.concatenate([expert_probs, policy_probs], axis=0)
            all_probs = np.expand_dims(all_probs, axis=1)

            feed_dict = {

                self.local_state: all_local,
                self.local_two_state: all_local_two,
                self.agent_stats: all_agent_stats,
                self.target_stats: all_target_stats,

                self.local_state_n: all_local_n,
                self.local_two_state_n: all_local_two_n,
                self.agent_stats_n: all_agent_stats_n,
                self.target_stats_n: all_target_stats_n,

                self.probs: all_probs,
                self.labels: labels,

                self.use_noise: [1]
            }

            if self.with_action:
                all_acts = np.concatenate([expert_acts, policy_acts], axis=0)
                all_acts = np.expand_dims(all_acts, axis=1)

                feed_dict[self.acts] = all_acts


            feed_dict[self.global_state] = all_global
            feed_dict[self.global_state_n] = all_global_n

            if self.use_vairl:
                loss, f, _, _, kl_loss = self.sess.run([self.loss, self.f, self.step, self.update_beta, self.kl_loss],
                                                       feed_dict=feed_dict)
            else:
                loss, f, disc, _ = self.sess.run([self.loss, self.f, self.discriminator, self.step],
                                                       feed_dict=feed_dict)

            losses.append(loss)

        # Update nomralization parameters
        for it in range(self.num_itr):

            expert_batch_idxs = random.sample(range(len(self.expert_traj['obs'])), self.batch_size)
            policy_batch_idxs = random.sample(range(len(self.buffer['obs'])), self.batch_size)

            expert_obs = [self.expert_traj['obs'][id] for id in expert_batch_idxs]
            policy_obs = [self.buffer['obs'][id] for id in policy_batch_idxs]

            expert_obs_n = [self.expert_traj['obs_n'][id] for id in expert_batch_idxs]
            policy_obs_n = [self.buffer['obs_n'][id] for id in policy_batch_idxs]

            expert_acts = [self.expert_traj['acts'][id] for id in expert_batch_idxs]
            policy_acts = [self.buffer['acts'][id] for id in policy_batch_idxs]

            e_states = self.obs_to_state(expert_obs)
            p_states = self.obs_to_state(policy_obs)

            all_global = np.concatenate([e_states[0], p_states[0]], axis=0)
            all_local = np.concatenate([e_states[1], p_states[1]], axis=0)
            all_local_two = np.concatenate([e_states[2], p_states[2]], axis=0)
            all_agent_stats = np.concatenate([e_states[3], p_states[3]], axis=0)
            all_target_stats = np.concatenate([e_states[4], p_states[4]], axis=0)

            e_states_n = self.obs_to_state(expert_obs_n)
            p_states_n = self.obs_to_state(policy_obs_n)

            all_global_n = np.concatenate([e_states_n[0], p_states_n[0]], axis=0)
            all_local_n = np.concatenate([e_states_n[1], p_states_n[1]], axis=0)
            all_local_two_n = np.concatenate([e_states_n[2], p_states_n[2]], axis=0)
            all_agent_stats_n = np.concatenate([e_states_n[3], p_states_n[3]], axis=0)
            all_target_stats_n = np.concatenate([e_states_n[4], p_states_n[4]], axis=0)

            expert_probs = []
            for (index, state) in enumerate(expert_obs):
                _, probs = self.select_action(state)
                expert_probs.append(probs[expert_acts[index]])

            policy_probs = []
            for (index, state) in enumerate(policy_obs):
                _, probs = self.select_action(state)
                policy_probs.append(probs[policy_acts[index]])

            expert_probs = np.asarray(expert_probs)
            policy_probs = np.asarray(policy_probs)

            probs = np.concatenate([expert_probs, policy_probs], axis=0)
            probs = np.expand_dims(probs, axis=1)

            feed_dict = {

                self.global_state: all_global,
                self.local_state: all_local,
                self.local_two_state: all_local_two,
                self.agent_stats: all_agent_stats,
                self.target_stats: all_target_stats,

                self.global_state_n: all_global_n,
                self.local_state_n: all_local_n,
                self.local_two_state_n: all_local_two_n,
                self.agent_stats_n: all_agent_stats_n,
                self.target_stats_n: all_target_stats_n,
            }

            if self.use_vairl:
                feed_dict[self.use_noise] = [0]

            if self.with_action:
                all_acts = np.concatenate([expert_acts, policy_acts], axis=0)
                all_acts = np.expand_dims(all_acts, axis=1)

                feed_dict[self.acts] = all_acts


            feed_dict[self.global_state] = all_global
            feed_dict[self.global_state_n] = all_global_n

            f = self.sess.run([self.f], feed_dict=feed_dict)
            f -= self.entropy_weight*np.log(probs)
            f = np.squeeze(f)
            self.push_reward(f)

        # Update Dynamic Running Stat
        if isinstance(self.r_norm, DynamicRunningStat):
            self.r_norm.reset()

        return np.mean(losses), 0

    # Eval without discriminator - only reward function
    def eval(self, obs, obs_n, acts=None, probs=None):

        states = self.obs_to_state(obs)

        feed_dict = {
            self.global_state: states[0],
            self.local_state: states[1],
            self.local_two_state: states[2],
            self.agent_stats: states[3],
            self.target_stats: states[4],

            self.use_noise: [0]
        }

        if self.with_action and acts is not None:
            acts = np.expand_dims(acts, axis=1)
            feed_dict[self.acts] = acts


        reward = self.sess.run([self.reward], feed_dict=feed_dict)
        if probs != None:
            reward -= self.entropy_weight*np.log(probs)
        
        # Normalize the reward
        #self.r_norm.push(reward[0][0])
        #reward = [[self.normalize_rewards(reward[0][0])]]
        #if self.r_norm.n == 0:
        #    reward = [[0]]


        return reward[0][0]

    # Eval with the discriminator - it returns an entropy regularized objective
    def eval_discriminator(self, obs, obs_n, probs, acts=None):
        states = self.obs_to_state(obs)
        states_n = self.obs_to_state(obs_n)

        probs = np.expand_dims(probs, axis=1)

        feed_dict = {
            self.global_state: states[0],
            self.local_state: states[1],
            self.local_two_state: states[2],
            self.agent_stats: states[3],
            self.target_stats: states[4],

            self.global_state_n: states[0],
            self.local_state_n: states_n[1],
            self.local_two_state_n: states_n[2],
            self.agent_stats_n: states_n[3],
            self.target_stats_n: states_n[4],

            self.use_noise: [0]
        }

        if self.with_action and acts is not None:
            acts = np.expand_dims(acts, axis=1)
            feed_dict[self.acts] = acts

        f = self.sess.run([self.f], feed_dict=feed_dict)
        f -= self.entropy_weight*np.log(probs)
        f = self.normalize_rewards(f)
        return f

    # Transform a DeepCrawl obs to state
    def obs_to_state(self, obs):

        global_batch = np.stack([np.asarray(state['global_in']) for state in obs])
        local_batch = np.stack([np.asarray(state['local_in']) for state in obs])
        local_two_batch = np.stack([np.asarray(state['local_in_two']) for state in obs])
        agent_stats_batch = np.stack([np.asarray(state['agent_stats']) for state in obs])
        target_stats_batch = np.stack([np.asarray(state['target_stats']) for state in obs])

        return global_batch, local_batch, local_two_batch, agent_stats_batch, target_stats_batch

    # For V-AIRL
    def make_beta_update(self):

        new_beta = tf.maximum(
            self.beta + self.alpha * (self.kl_loss - self.mutual_information), eps
        )
        with tf.control_dependencies([self.step]):
            self.update_beta = tf.compat.v1.assign(self.beta, new_beta)

    # Normalize the reward for each frame of the sequence
    def push_reward(self, rewards):
        for r in rewards:
            self.r_norm.push(r)

    def normalize_rewards(self, rewards):
        rewards -= self.r_norm.mean
        rewards /= (self.r_norm.std + 1e-12)
        rewards *= 0.05

        return rewards

    # Select action from the policy and fetch the probability distribution over the action space
    def select_action(self, state):

        act, _, probs = self.policy.eval([state])

        return (act, probs[0])

    # Update demonstrations
    def set_demonstrations(self, demonstrations, validations):
        self.expert_traj = demonstrations

        if validations is not None:
            self.validation_traj = validations

    # Create the replay buffer with which train the discriminator
    def create_buffer(self):
        self.buffer['obs'] = []
        self.buffer['obs_n'] = []
        self.buffer['acts'] = []

    # Add a transition to the buffer
    def add_to_buffer(self, obs, obs_n, acts, buffer_length=100000):

        if len(self.buffer['obs']) >= buffer_length:
            random_index = np.random.randint(0, len(self.buffer['obs']))
            del self.buffer['obs'][random_index]
            del self.buffer['obs_n'][random_index]
            del self.buffer['acts'][random_index]

        self.buffer['obs'].append(obs)
        self.buffer['obs_n'].append(obs_n)
        self.buffer['acts'].append(acts)


    # Create and return some demonstrations [(states, actions, frames)]. The order of the demonstrations must be from
    # best to worst. The number of demonstrations is given by the user
    def create_demonstrations(self, env, save_demonstrations=True, inference=False, verbose=False,
                              with_policy=False, num_episode=31, max_timestep=20):
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
                try:
                    # Input the action and save the new state and action
                    step += 1
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
                        action, probs = self.select_action(state)
                    print(action)
                    state_n, done, reward = env.execute(action)

                    cum_reward += reward
                    if not with_policy:
                        action = env.command_to_action(action)
                    # If inference is true, print the reward
                    if inference:
                        _, probs = self.select_action(state)
                        reward = self.eval([state], [state_n], [action])
                        # print('Discriminator probability: ' + str(disc))
                        print('Unnormalize reward: ' + str(reward))
                        reward = self.normalize_rewards(reward)
                        print('Normalize reward: ' + str(reward))
                        print('Probability of state space: ')
                        print(probs)
                    state = state_n
                    states.append(state)
                    actions.append(action)
                    if step >= max_timestep:
                        done = True
                except Exception as e:
                    print(e)
                    continue

            if not inference:
                y = None

                print('Demonstration number: ' + str(episode))
                if with_policy:
                    if episode < num_episode:
                        print(state_n['target_stats'][0])
                        if True:
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

                    if with_policy:
                        if episode > num_episode - 1:
                            y = input('Do you want to save this demonstration as validation? [y/n] ')
                        else:
                            y = 'n'
                    else:
                        y = input('Do you want to save this demonstration as validation? [y/n] ')

                    if y == 'y':
                        val_traj['obs'].extend(np.array(states[:-1]))
                        val_traj['obs_n'].extend(np.array(states[1:]))
                        val_traj['acts'].extend(np.array(actions))
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

        return expert_traj, val_traj

    # Save demonstrations dict to file
    @staticmethod
    def save_demonstrations(demonstrations, validations=None, name='dems_potions.pkl'):
        with open('reward_model/dems/' + name, 'wb') as f:
            pickle.dump(demonstrations, f, pickle.HIGHEST_PROTOCOL)
        if validations is not None:
            with open('reward_model/dems/vals.pkl', 'wb') as f:
                pickle.dump(validations, f, pickle.HIGHEST_PROTOCOL)

    # Load demonstrations from file
    def load_demonstrations(self, name='dems.pkl'):
        with open('reward_model/dems/' + name, 'rb') as f:
            expert_traj = pickle.load(f)

        with open('reward_model/dems/vals.pkl', 'rb') as f:
            val_traj = pickle.load(f)

        self.set_demonstrations(expert_traj, val_traj)

        return expert_traj, val_traj

    # Save the entire model
    def save_model(self, name=None):
        self.saver.save(self.sess, 'reward_model/models/{}'.format(name))
        return

    # Load entire model
    def load_model(self, name=None):
        self.saver = tf.compat.v1.train.import_meta_graph('reward_model/models/' + name + '.meta')
        self.saver.restore(self.sess, 'reward_model/models/' + name)
        return
