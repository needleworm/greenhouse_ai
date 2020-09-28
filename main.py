"""
    Author : Byunghyun Ban
    SBIE @ KAIST
    needleworm@kaist.ac.kr
    latest modification :
        2017.06.12.
"""

import csv
import tensorflow as tf
import random
import numpy as np
import os
import agent
import vinyl_house
from collections import deque

logs_dir = "logs"

MAX_EPISODE = 100000000
RANDOM_SEED = 1234
BUFFER_SIZE = 10000
MAX_EP_STEPS = 1000
learning_rate = 0.0001
tau = 0.001

stride = 1
stddev = 0.02

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("device", "/gpu:0", "device : /cpu:0, /gpu:0, /gpu:1. Default is /gpu:0.")
tf.flags.DEFINE_bool("Train", "True", "mode : train, test. Default is train")
tf.flags.DEFINE_bool("reset", "False", "mode : True or False. Default is train")
tf.flags.DEFINE_integer("num_actions", "8", "number of actions. Default is 8.")
tf.flags.DEFINE_integer("num_observation", "7", "number of actions. Default is 210.")
tf.flags.DEFINE_float("gamma", "0.95", "discount factor of Q learning. Default is 0.9")
tf.flags.DEFINE_float("random_init", "0.00000001", "initial probability for randomly sampling action. Default is 1.0")
tf.flags.DEFINE_float("random_final", "0.0000001", "final probability for randomly sampling action. Default is 0.1")
tf.flags.DEFINE_float("epsilon_decay", "0.1", "epsilon decay rate. Default is 0.95")
tf.flags.DEFINE_integer("epsilon_decay_step", "9", "epsilon decay step. Default is 10")
tf.flags.DEFINE_integer("batch_size", "256", "mini batch size. Default is 2")
tf.flags.DEFINE_integer("mini_batch_size", "256", "mini batch size. Default is 64")

if FLAGS.reset:
    os.popen("rm -rf " + logs_dir)
    os.popen("mkdir " + logs_dir)
    os.popen("mkdir " + logs_dir + "/loss")
    os.popen("mkdir " + logs_dir + "/score")
    os.popen("mkdir " + logs_dir + "/q")


class Actor(object):
    def __init__(self):
        self.observation = tf.placeholder(tf.float32, [None, FLAGS.num_observation])
        self.observation_target = tf.placeholder(tf.float32, [None, FLAGS.num_observation])

        with tf.variable_scope("Actor") as t:
            self.network = agent.Actor(FLAGS.num_actions, FLAGS.num_observation)
            self.out, _ = self.network.create_graph(self.observation)
            self.network_params = tf.trainable_variables()

            t.reuse_variables()
            self.network_target = agent.Actor(FLAGS.num_actions, FLAGS.num_observation)
            self.target_out, _ = self.network_target.create_graph(self.observation_target)
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - tau))
                for i in range(len(self.target_network_params))]

        self.action_gradient = tf.placeholder(tf.float32, [None, FLAGS.num_actions])

        self.actor_gradients = tf.gradients(self.out, self.network_params, -self.action_gradient)
        self.optimize = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def action(self, P, feed, eps):
        act_values = P.eval(feed_dict=feed)
        x, y = act_values.shape
        action = np.zeros((act_values.shape), dtype=np.bool)
        if random.random() <= eps:
            for j in range(x):
                for i in range(y):
                    if random.random() > 0.7:
                        action[j, i] = True
        else:
            for j in range(x):
                for i in range(y):
                    if act_values[j, i] > 0.5:
                        action[j, i] = True
        return action


class Critic(object):
    def __init__(self, num_actor_vars):
        self.observation = tf.placeholder(tf.float32, [None, FLAGS.num_observation])
        self.observation_target = tf.placeholder(tf.float32, [None, FLAGS.num_observation])
        self.action_target = tf.placeholder(tf.float32, [None, FLAGS.num_actions])
        self.actions = tf.placeholder(tf.float32, [None, FLAGS.num_actions])

        with tf.variable_scope("critic") as c:
            self.network = agent.Critic(FLAGS.num_actions, FLAGS.num_observation)
            self.out, _ = self.network.create_graph(self.observation, self.actions)
            self.network_params = tf.trainable_variables()[num_actor_vars:]
            c.reuse_variables()
            self.network_target = agent.Critic(FLAGS.num_actions, FLAGS.num_observation)
            self.target_out, _ = self.network_target.create_graph(self.observation_target, self.action_target)
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], tau) + tf.multiply(self.target_network_params[i], 1. - tau))
                for i in range(len(self.target_network_params))]

        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        self.loss = tf.reduce_mean(tf.squared_difference(self.predicted_q_value, self.out))
        self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.action_grads = tf.gradients(self.out, self.actions)

    def predict(self, P, feed):
        return P.eval(feed_dict=feed)


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.deque.clear()
        self.count = 0


def train(env):
    ###############################  GRAPH PART  ###############################
    print("Graph Initialization...")
    with tf.device(FLAGS.device):
        with tf.variable_scope("model", reuse=None):
            m_actor = Actor()
            m_critic = Critic(m_actor.num_trainable_vars)
    print("Done")

    ##############################  Summary Part  ##############################
    print("Setting up summary op...")
    loss_ph = tf.placeholder(dtype=tf.float32)
    loss_summary_op = tf.summary.scalar("loss", loss_ph)
    score_ph = tf.placeholder(dtype=tf.float32)
    score_summary_op = tf.summary.scalar("score", score_ph)
    q_ph = tf.placeholder(dtype=tf.float32)
    q_summary_op = tf.summary.scalar("Q value", q_ph)

    train_loss_writer = tf.summary.FileWriter(logs_dir + '/loss/', max_queue=2)
    train_score_writer = tf.summary.FileWriter(logs_dir + '/score/', max_queue=2)
    train_q_writer = tf.summary.FileWriter(logs_dir + '/q/', max_queue=2)
    print("Done")

    ############################  Model Save Part  #############################
    print("Setting up Saver...")
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    print("Done")

    ################################  Session Part  ################################
    print("Session Initialization...")
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=sess_config)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("model restored...")
    else:
        sess.run(tf.global_variables_initializer())

    # Some initial local variables
    eps = FLAGS.random_init


    global_step = 0
    learning_finished = False
    rewards= None
    # Score cache
    score_queue = []

    exp_pointer = -1
    decay = 0

    sess.run([m_actor.update_target_network_params, m_critic.update_target_network_params])
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
    # The episode loop
    for i_episode in range(0, 100):
        previous, operations, reward, _ = env.reset()
        """
        env.reset()을 하면 모든 변수가 초기화됨.
        reset()이나 step(action)을 실행하면
        상태, 작동중인 액션, 점수, Done 네가지가 리턴됨.
        """
        done = False
        REWARDS = 0
        q_ave_max = 0

        if decay % FLAGS.epsilon_decay_step == 0 and eps > FLAGS.random_final:
            eps *= FLAGS.epsilon_decay
        previousobs = previous

        # The step loop
        while not done:
            """
            Environment가 env.step()에서 Done == True을 리턴하면 while문이 중지됨.
            게임 한 판 (한 에피소드)가 종료된 것.
            """
            action = m_actor.action(m_actor.out, {m_actor.observation: np.expand_dims(previous, 0)}, eps)

            obs, operations, reward, done = env.step(action)
            """
            action을 env.step()에 집어넣어 줌.
            비닐하우스는 액션을 받아먹고, 액션을 취하여 내부변수를 변화시킴.
            obs : 변화된 이후의 내부상태
            operations : 현재 작동중인 액션
            reward : 이번 스텝에서의 점수
            done : 게임이 끝났는지 (농장이 파멸했는지 여부. True / False)
            """



            observation = obs - previousobs

            REWARDS += reward
            """
            매 step별로, 스텝 점수가 리턴됨.
            이 점수를 차곡 차곡 쌓으면 됨.
            일반적인 상황에서 이 점수는 틱당 1점.
            게임이 터지면 -1점 리턴
            최종적으로는 REWARD를 보면 얘가 몇시간 제어했는지 알 수 있음.
            ex) 10턴째 게임 터짐
                REWARDS += reward 동작을 10번 수행
                0~8번째에서는 reward == 1이므로 REWARDS == 9
                9번째에서 reward == -1이므로 REWARDS == 8
                --> 첫 시도는 계산하지 않음
                --> 마지막 시도는 제어 실패로 이어짐
                --> 제어에 성공한 시간은 8시간

                REWARDS가 제어성공시간이 됨됨
            """

            replay_buffer.add(np.reshape(previous, (FLAGS.num_observation,)), np.reshape(action, (FLAGS.num_actions,)), reward, done, np.reshape(observation, (FLAGS.num_observation,)))

            if replay_buffer.size() > FLAGS.mini_batch_size:
                prev_obs_batch, act_batch, rwd_batch, done_batch, obs_batch = replay_buffer.sample_batch(FLAGS.mini_batch_size)

                actor_action = m_actor.action(m_actor.target_out, {m_actor.observation_target: obs_batch}, eps)
                target_q = sess.run(m_critic.target_out, feed_dict={m_critic.observation_target: obs_batch, m_critic.action_target: actor_action})

                y_i = []
                for k in range(FLAGS.mini_batch_size):
                    if done_batch[k]:
                        y_i.append(rwd_batch[k])
                    else:
                        y_i.append(rwd_batch[k] + FLAGS.gamma * target_q[k])

                predicted_q, _ = sess.run([m_critic.out, m_critic.optimize], feed_dict = {m_critic.observation : prev_obs_batch, m_critic.actions: act_batch, m_critic.predicted_q_value: np.reshape(y_i, (FLAGS.mini_batch_size, 1))})

                q_ave_max += np.amax(predicted_q)

                act_out = sess.run(m_actor.out, feed_dict={m_actor.observation : prev_obs_batch})
                grads = sess.run(m_critic.action_grads, feed_dict={m_critic.observation : prev_obs_batch, m_critic.actions: act_out})
                sess.run(m_actor.optimize, feed_dict={m_actor.observation: prev_obs_batch, m_actor.action_gradient: grads[0]})

                m_actor.update_target_network_params
                m_critic.update_target_network_params

            previous = observation
            previousobs = obs

            if done:
                score_str = sess.run(score_summary_op, feed_dict={score_ph: REWARDS})
                train_score_writer.add_summary(score_str, i_episode)
                q_str = sess.run(q_summary_op, feed_dict={q_ph: q_ave_max})
                train_q_writer.add_summary(q_str, i_episode)


        #print("\n====== Episode " + str(i_episode) + " ended with score " + str(reward))
        score_queue.append(REWARDS)


        if i_episode % 100000 == 0:
            saver.save(sess, logs_dir + "/model.ckpt", i_episode)

        if learning_finished:
            print("Testing !!!")

    print(i_episode + REWARDS)


def main():
    env = vinyl_house.Vinylhouse(allowed_minimum_temparature=10,
                            allowed_maximum_temparature=100,
                            allowed_minimum_humidity_air=10,
                            allowed_maximum_humidity_air=90,
                            allowed_minimum_humidity_ground=30,
                            allowed_maximum_humidity_ground=80,
                            plants_harvested_with_height=70,
                            current_temparature=21,
                            current_humidity_air=50,
                            current_humidity_ground=50,
                            current_plant_height=0)
    train(env)

def action(obs, ops):
    """
    observations[0] = self.current_temparature
    observations[1] = self.current_humidity_air
    observations[2] = self.current_humidity_ground
    observations[3] = self.current_plant_height
    observations[4] = self.pesticide_density
    observations[5] = float(self.is_human) * 100
    observations[6] = float(self.insect_detected) * 100

        if action == 0:
            self.fan_on()
        if action == 1:
            self.curtain_open()
        if action == 2:
            self.water_inject_inside()
        if action == 3:
            self.water_inject_outside()
        if action == 4:
            self.pesticide_injection()
        if action == 5:
            self.LED_on
        if action == 6:
            self.nutrients_spray()
        if action == 7:
            self.harvest()
    """
    action = np.zeros(8, dtype=np.uint8)

    if obs[0] > 30:
        # 온도 50도 넘으면 선풍기 커튼 1과 LED 제어
        if ops[0] is 0:
            action[0] = 1
        if ops[1] is 0:
            action[1] = 1
        if ops[5] is 1:
            action[5] = 1

    if obs[0] < 30:
        # 온도 30도 이하로 떨어지면 꺼줌
        if ops[0] is 1:
            action[0] = 0
        if ops[1] is 1:
            action[1] = 0
        if ops[5] is 0:
            action[5] = 1

    # 공기 중 습도 제어
    if obs[1] > 60:
        if ops[1] is 0:
            action[1] = 1
        if ops[2] is 1:
            action[2] = 0

    if obs[1] < 60:
        if ops[1] is 1:
            action[1] = 0

        if ops[2] is 0:
            action[2] = 1

    # ground humidity
    if obs[2] < 60:
        if ops[2] is 0:
            action[2] = 1
        if ops[1] == 1:
            action[1] = 0
        if ops[0] is 1:
            action[0] = 0
        if ops[3] is 0:
            action[3] = 1

    if obs[2] > 60:
        if ops[2] is 1:
            action[2] = 0

    # insect control
    # 사람 없으면 약 치고 있으면 벌레 있어도 치지 마
    if obs[6] == 100 and obs[5] != 100:
        if ops[5] is 0:
            action[5] = 1
        if ops[6] is 0:
            action[6] = 1
    if obs[6] == 0:
        if ops[5] is 1:
            action[5] = 0
        if ops[6] is 1:
            action[6] = 0

    if obs[3] < 80:
        if ops[7] is 1:
            action[7] = 0
    else:
        if ops[7] is 0:
            action[7] = 1

    return action

f = open('data.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
writerow = 0

def testcase():
    env = vinyl_house.Vinylhouse(allowed_minimum_temparature=10,
                            allowed_maximum_temparature=100,
                            allowed_minimum_humidity_air=10,
                            allowed_maximum_humidity_air=90,
                            allowed_minimum_humidity_ground=30,
                            allowed_maximum_humidity_ground=80,
                            plants_harvested_with_height=70,
                            current_temparature=21,
                            current_humidity_air=50,
                            current_humidity_ground=50,
                            current_plant_height=0)
    avg = 0
    for i in range(0, 5000):
        REWARDS = 0
        obs, ops, rwd, done = env.reset()
        while not done:
            act = action(obs, ops)
            obs, ops, rwd, done = env.step(act)
            REWARDS += rwd

        f.write(str(REWARDS) + '\n')
        avg += REWARDS
    print(avg / 5000)


testcase()

f.close()
