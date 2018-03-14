# Yerbol Aussat
# CS 686: Artificial Intelligence
# used an example from https://github.com/MorvanZhou/ 

import numpy as np
import random
import gym
import tflearn
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env = env.unwrapped


class dqn:
	# initializing dqn
    def __init__(self, n_actions, n_features):
        self.n_actions = n_actions
        self.n_features = n_features
        #parametrs
        self.gamma = 0.99
        self.buffer_size = 1000
        self.batch_size = 1
        self.epsilon = 0.05
        self.learning_rate = 0.1
        # buffer [s, a, r, s_prime]
        self.buffer = np.zeros((self.buffer_size, n_features * 2 + 2))
        self.build_networks() # target and evaluation
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.total_samples= 0
        

    def build_networks(self):
        # evaluation network
        
        self.s = tf.placeholder(tf.float32, [None, 4])
        self.q_target = tf.placeholder(tf.float32, [None, 2]) 
        
        with tf.variable_scope('evaluation_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first hidden layer
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [4, 10], initializer=tf.random_normal_initializer(0., 0.3), collections=c_names)
                b1 = tf.get_variable('b1', [1, 10], initializer=tf.constant_initializer(0.1), collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
                
            # second hidden layer
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w1', [10, 10], initializer=tf.random_normal_initializer(0., 0.3), collections=c_names)
                b2 = tf.get_variable('b1', [1, 10], initializer=tf.constant_initializer(0.1), collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            # output layer
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w2', [10, 2], initializer=tf.random_normal_initializer(0., 0.3), collections=c_names)
                b3 = tf.get_variable('b2', [1, 2], initializer=tf.constant_initializer(0.1), collections=c_names)
                self.q_eval = tf.matmul(l2, w3) + b3

        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        self.train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss)
            

        # target_network
        self.s_prime = tf.placeholder(tf.float32, [None, self.n_features], name='s_prime')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first hidden layer
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [4, 10], initializer=tf.random_normal_initializer(0., 0.3), collections=c_names)
                b1 = tf.get_variable('b1', [1, 10], initializer=tf.constant_initializer(0.1), collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
                
            # second hidden layer
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w1', [10, 10], initializer=tf.random_normal_initializer(0., 0.3), collections=c_names)
                b2 = tf.get_variable('b1', [1, 10], initializer=tf.constant_initializer(0.1), collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            # output layer
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w2', [10, 2], initializer=tf.random_normal_initializer(0., 0.3), collections=c_names)
                b3 = tf.get_variable('b2', [1, 2], initializer=tf.constant_initializer(0.1), collections=c_names)
                self.q_next = tf.matmul(l2, w3) + b3

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, 2)
        else:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
    	return action
    	
    def getQ(self, observation):
    	observation = observation[np.newaxis, :]
    	actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
    	return actions_value	
    
    def replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

    def learn(self, prev_obs, action, reward, new_obs):

    	prev_obs = prev_obs[np.newaxis, :]
    	new_obs = new_obs[np.newaxis, :]

        q_next, q_eval = self.sess.run([self.q_next, self.q_eval],
            feed_dict={self.s_prime: new_obs, self.s: prev_obs})

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()        
        q_target[0, action] = reward + self.gamma * np.max(q_next) 

        # train eval network:  feed observations and q_target values
        self.sess.run([self.train],	feed_dict={self.s: prev_obs, self.q_target: q_target})

deep_q_n = dqn(2, 4)
total_steps = 0

#recording all rewards
rewards = []
Qs = []

for episode in range(1000):
	prev_obs = env.reset()
	episode_reward = 0
	steps = 0
	while True and steps < 500:
		steps+=1
        #env.render()
		action = deep_q_n.choose_action(prev_obs)
		new_obs, reward, done, info = env.step(action)
		episode_reward += reward
		deep_q_n.learn(prev_obs, action, reward, new_obs)

		if done:
			print('ep: ', episode,
				  'total_r: ', episode_reward)
			rewards.append(episode_reward)
			
			Q = deep_q_n.getQ(new_obs).max()
			Qs.append(Q)
			break

		prev_obs = new_obs
 		total_steps += 1
		
	if episode % 2 == 0:
		deep_q_n.replace_target_params()
		
plt.plot(range(1000), Qs)
plt.ylabel('Discounted total reward')
plt.xlabel('episodes')
plt.show()