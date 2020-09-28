"""
    Author : Byunghyun Ban
    SBIE @ KAIST
    needleworm@kaist.ac.kr
    latest modification :
        2017.06.5.
"""

__author__ = 'BHBAN'

import tensorflow as tf

stddev=0.02


class Actor(object):
    def __init__(self, num_actions, num_observations):
        self.Actor_Graph = Actor_Graph(num_actions, num_observations)

    def create_graph(self, observation):
        logits = self.Actor_Graph.graph(observation)
        return logits[-1], logits


class Actor_Graph(object):
    def __init__(self, num_actions, num_observations):
        # Encoder
        self.FNN1_shape = [num_observations, 128]
        self.kernel1 = tf.get_variable("index_1_W", initializer=tf.truncated_normal(self.FNN1_shape, stddev=stddev))
        self.bias1 = tf.get_variable("index_1_B", initializer=tf.constant(0.1, shape=[self.FNN1_shape[-1]]))

        self.FNN2_shape = [128, 128]
        self.kernel2 = tf.get_variable("index_2_W", initializer=tf.truncated_normal(self.FNN2_shape, stddev=stddev))
        self.bias2 = tf.get_variable("index_2_B", initializer=tf.constant(0.1, shape=[self.FNN2_shape[-1]]))

        self.FNN3_shape = [128, 32]
        self.kernel3 = tf.get_variable("index_3_W", initializer=tf.truncated_normal(self.FNN3_shape, stddev=stddev))
        self.bias3 = tf.get_variable("index_3_B", initializer=tf.constant(0.1, shape=[self.FNN3_shape[-1]]))

        self.FNN4_shape = [32, num_actions]
        self.kernel4 = tf.get_variable("index_4_W", initializer=tf.truncated_normal(self.FNN4_shape, stddev=stddev))
        self.bias4 = tf.get_variable("index_4_B", initializer=tf.constant(0.1, shape=[self.FNN4_shape[-1]]))

    def graph(self, observation):
        net = []
        net.append(observation)

        # First FNN Layer
        W1 = tf.matmul(tf.reshape(observation, [-1, self.FNN1_shape[0]]) , self.kernel1)
        W1 = tf.nn.bias_add(W1, self.bias1)
        R1 = tf.nn.relu(W1, name="ReLu1")
        net.append(R1)

        # Second FNN layer
        W2 = tf.matmul(R1, self.kernel2)
        W2 = tf.nn.bias_add(W2, self.bias2)
        R2 = tf.nn.relu(W2, name="ReLu2")
        net.append(R2)

        # Third FNN Layer
        W3 = tf.matmul(R2, self.kernel3)
        W3 = tf.nn.bias_add(W3, self.bias3)
        R3 = tf.nn.relu(W3, name="ReLu3")
        net.append(R3)

        # Fourth FNN Layer
        W4 = tf.matmul(R3, self.kernel4)
        W4 = tf.nn.bias_add(W4, self.bias4)

        # Readout
        Out= tf.nn.softmax(W4)
        net.append(Out)

        return net


class Critic(object):
    def __init__(self, num_actions, num_observations):
        self.Critic_Graph = Critic_Graph(num_actions, num_observations)

    def create_graph(self, observation, actions):
        logits = self.Critic_Graph.graph(observation, actions)
        return logits[-1], logits


class Critic_Graph(object):
    def __init__(self, num_actions, num_observations):
        # Encoder
        self.FNN1_shape = [num_observations, 256]
        self.kernel1 = tf.get_variable("index_1_W", initializer=tf.truncated_normal(self.FNN1_shape, stddev=stddev))
        self.bias1 = tf.get_variable("index_1_B", initializer=tf.constant(0.1, shape=[self.FNN1_shape[-1]]))

        self.FNN2_shape = [256, 128]
        self.kernel2 = tf.get_variable("index_2_W", initializer=tf.truncated_normal(self.FNN2_shape, stddev=stddev))
        self.bias2 = tf.get_variable("index_2_B", initializer=tf.constant(0.1, shape=[self.FNN2_shape[-1]]))

        self.FNN3_shape = [num_actions, 128]
        self.kernel3 = tf.get_variable("index_3_W", initializer=tf.truncated_normal(self.FNN3_shape, stddev=stddev))

        self.FNN4_shape = [128, 1]
        self.kernel4 = tf.get_variable("index_4_W", initializer=tf.truncated_normal(self.FNN4_shape, stddev=stddev))
        self.bias4 = tf.get_variable("index_4_B", initializer=tf.constant(0.1, shape=[self.FNN4_shape[-1]]))

    def graph(self, inputs, actions):
        net = []
        net.append(inputs)

        # First FNN Layer
        W1 = tf.matmul(tf.reshape(inputs, [-1, self.FNN1_shape[0]]) , self.kernel1)
        W1 = tf.nn.bias_add(W1, self.bias1)
        R1 = tf.nn.relu(W1, name="ReLu1")

        S1 = tf.matmul(R1, self.kernel2) + tf.matmul(actions, self.kernel3)
        S1 = tf.nn.bias_add(S1, self.bias2)
        R2 = tf.nn.relu(S1)
        net.append(R2)
        W2 = tf.matmul(R2, self.kernel4)
        W2 = tf.nn.bias_add(W2, self.bias4)

        out = W2
        net.append(out)
        return net
