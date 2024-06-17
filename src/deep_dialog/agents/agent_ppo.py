'''
Created on Oct 30, 2017

An DQN Agent modified for DDQ Agent

Some methods are not consistent with super class Agent.

@author: Baolin Peng
'''
import logging
import os.path
import random, copy, json
import cPickle as pickle
import sys

import numpy as np
from collections import namedtuple, deque

from deep_dialog import dialog_config

from agent import Agent
from deep_dialog.qlearning import DQN
from deep_dialog.qlearning import Value, MultiDiscretePolicy

import torch
import torch.optim as optim


import tensorflow as tf
import torch.nn.functional as F

DEVICE = torch.device('cpu')

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'term'))


class AgentPPO(Agent):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.feasible_actions = dialog_config.feasible_actions
        self.num_actions = len(self.feasible_actions)

        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']

        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 5000)
        self.experience_replay_pool = deque(
            maxlen=self.experience_replay_pool_size)  # experience replay pool <s_t, a_t, r_t, s_t+1>
        self.experience_replay_pool_from_model = deque(
            maxlen=self.experience_replay_pool_size)  # experience replay pool <s_t, a_t, r_t, s_t+1>
        self.running_expereince_pool = None # hold experience from both user and world model

        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.gamma = params.get('gamma', 0.9)
        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)

        self.max_turn = params['max_turn'] + 5
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn

        self.tau = 0.95
        self.ppo_epsilon = 0.2
        self.ppo_value = Value(self.state_dimension, 30).to(DEVICE)
        self.policy = MultiDiscretePolicy(self.state_dimension, 60, self.num_actions).to(DEVICE)
        self.policy_optim = optim.RMSprop(self.policy.parameters(), lr=0.0001)
        self.value_optim = optim.Adam(self.ppo_value.parameters(), lr=0.00005)

        # self.dqn = DQN(self.state_dimension, self.hidden_size, self.num_actions).to(DEVICE)
        # self.target_dqn = DQN(self.state_dimension, self.hidden_size, self.num_actions).to(DEVICE)
        # self.target_dqn.load_state_dict(self.dqn.state_dict())
        # self.target_dqn.eval()

        # self.optimizer = optim.RMSprop(self.dqn.parameters(), lr=1e-3)

        self.cur_bellman_err = 0

        # Prediction Mode: load trained DQN model
        if params['trained_model_path'] != None:
            self.load(params['trained_model_path'])
            self.predict_mode = True
            self.warm_start = 2

    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """

        self.current_slot_id = 0
        self.phase = 0
        self.request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']

    def state_to_action(self, state):
        """ DQN: Input state, output action """
        # self.state['turn'] += 2
        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation)
        if self.warm_start == 1:
            act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        else:
            act_slot_response = copy.deepcopy(self.feasible_actions[self.action[0]])

        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}

    def prepare_state_representation(self, state):
        """ Create the representation for each state """

        user_action = state['user_action']
        current_slots = state['current_slots']
        kb_results_dict = state['kb_results_dict']
        agent_last = state['agent_action']

        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        user_act_rep = np.zeros((1, self.act_cardinality))
        user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0

        ########################################################################
        #     Create bag of inform slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            user_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ########################################################################
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            current_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent act
        ########################################################################
        agent_act_rep = np.zeros((1, self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

        ########################################################################
        #   Encode last agent inform slots
        ########################################################################
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['inform_slots'].keys():
                agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent request slots
        ########################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

        # turn_rep = np.zeros((1,1)) + state['turn'] / 10.
        turn_rep = np.zeros((1, 1))

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        # ########################################################################
        # #   Representation of KB results (scaled counts)
        # ########################################################################
        # kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        # for slot in kb_results_dict:
        #     if slot in self.slot_set:
        #         kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.
        #
        # ########################################################################
        # #   Representation of KB results (binary)
        # ########################################################################
        # kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum( kb_results_dict['matching_all_constraints'] > 0.)
        # for slot in kb_results_dict:
        #     if slot in self.slot_set:
        #         kb_binary_rep[0, self.slot_set[slot]] = np.sum( kb_results_dict[slot] > 0.)

        kb_count_rep = np.zeros((1, self.slot_cardinality + 1))

        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1))

        self.final_representation = np.hstack(
            [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
             agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])
        return self.final_representation

    def run_policy(self, representation):
        """ epsilon-greedy policy """

        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                    self.warm_start = 2
                return self.rule_policy()
            else:
                return self.PPO_policy(representation)

    def rule_policy(self):
        """ Rule Policy """

        act_slot_response = {}

        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"},
                                 'request_slots': {}}
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}}

        return self.action_index(act_slot_response)

    def PPO_policy(self, state_representation):
        """ Return action from DQN"""

        with torch.no_grad():
            state_representation = torch.FloatTensor(state_representation)
            state_representation = state_representation.squeeze(0)
            # with open("/home/x/P/DDQ/mylog/final/s_p.txt", "w") as file:
            #     file.write(str(state_representation))
            # print("s.shape::::{}".format(state_representation.shape))
            # print("s.type::::{}".format(type(state_representation)))
            action = self.policy.select_action(torch.FloatTensor(state_representation))
            # print "action:::{}".format(action)
            # action = [np.argmax(i) for i in action]
            action = torch.argmax(action, dim=-1)
            action = torch.unsqueeze(action, 0)
            # action = torch.Tensor(action_temp)
            # print "action:::{}".format(action)
            # print "type:::{}".format(type(action))
        return action

    def action_index(self, act_slot_response):
        """ Return the index of action """

        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        print act_slot_response
        raise Exception("action index not found")
        return None

    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over, st_user, from_model=False):
        """ Register feedback from either environment or world model, to be stored as future training data """

        state_t_rep = self.prepare_state_representation(s_t)
        action_t = self.action
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        st_user = self.prepare_state_representation(s_tplus1)
        training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over, st_user)

        if self.predict_mode == False:  # Training Mode
            if self.warm_start == 1:
                self.experience_replay_pool.append(training_example)
        else:  # Prediction Mode
            if not from_model:
                self.experience_replay_pool.append(training_example)
            else:
                self.experience_replay_pool_from_model.append(training_example)

    def sample_from_buffer(self, batch_size):
        """Sample batch size examples from experience buffer and convert it to torch readable format"""
        # type: (int, ) -> Transition

        batch = [random.choice(self.running_expereince_pool) for i in xrange(batch_size)]
        np_batch = []
        for x in range(len(Transition._fields)):
            v = []
            for i in xrange(batch_size):
                v.append(batch[i][x])
            np_batch.append(np.vstack(v))

        return Transition(*np_batch)

    def print_log(self, variable_name, variable):
        base_path = "/home/x/P/DDQ/mylog/ppo/"
        variable_path = variable_name + ".txt"
        real_path = os.path.join(base_path, variable_path)
        with open(real_path, "w") as file:
            file.write(str(variable))
        print "{}.shape::::{}".format(variable_name, variable.shape)
        print "{}.type::::{}".format(variable_name, type(variable))
        print "-----------{}--saved----------------".format(variable_name)
        return

    def est_adv(self, r, v, mask):
        batchsz = v.size(0)
        # v_target = torch.Tensor(batchsz).to(device=DEVICE)
        # delta = torch.Tensor(batchsz).to(device=DEVICE)
        # A_sa = torch.Tensor(batchsz).to(device=DEVICE)
        v_target = torch.Tensor(batchsz)
        delta = torch.Tensor(batchsz)
        A_sa = torch.Tensor(batchsz)
        prev_v_target = 0
        prev_v = 0
        prev_A_sa = 0
        for t in reversed(range(batchsz)):
            v_target[t] = r[t] + self.gamma * prev_v_target * mask[t]
            delta[t] = r[t] + self.gamma * prev_v * mask[t] - v[t]
            A_sa[t] = delta[t] + self.gamma * self.tau * prev_A_sa * mask[t]
            prev_v_target = v_target[t]
            prev_v = v[t]
            prev_A_sa = A_sa[t]
        A_sa = (A_sa - A_sa.mean()) / A_sa.std()
        return A_sa, v_target

    def train(self, batch_size=1, num_batches=100, epoch=500):
        """ Train PPO with experience buffer that comes from both user and world model interaction."""
        # print "batch_size:::{}".format(batch_size)
        # sys.exit()
        """real batch_size = 16"""
        """real_num_batch=1"""
        print "epoch::::{}".format(epoch)
        torch.set_printoptions(threshold=np.inf)
        self.running_expereince_pool = list(self.experience_replay_pool) + list(self.experience_replay_pool_from_model)
        b_z = len(self.running_expereince_pool)
        print "b_z::::{}".format(b_z)
        batch = self.sample_from_buffer(b_z)
        # self.print_log('batch', batch)
        s = torch.FloatTensor(batch.state)
        self.print_log('s', s)
        a = torch.tensor(batch.action).numpy()
        a_one_hot = tf.one_hot(a, depth=29).numpy()
        a = torch.from_numpy(a_one_hot)
        a = a.squeeze(dim=1)
        self.print_log('a', a)
        r = torch.tensor(batch.reward)
        r = r.squeeze(dim=1)
        self.print_log('r', r)

        v = self.ppo_value(s).squeeze(-1).detach()
        self.print_log('v', v)

        log_pi_old_sa = self.policy.get_log_prob(s, a).detach()
        self.print_log('log_pi_old_sa', log_pi_old_sa)

        mask = torch.zeros_like(r)
        mask[r == -1] = 1
        self.print_log('mask', mask)

        A_sa, v_target = self.est_adv(r, v, mask)
        self.print_log('A_sa', A_sa)
        self.print_log('v_target', v_target)

        for i in range(5):
            perm = torch.randperm(b_z)
            v_target_shuf, A_sa_shuf, s_shuf, a_shuf, log_pi_old_sa_shuf = v_target[perm], A_sa[perm], s[perm], a[perm], \
                                                                           log_pi_old_sa[perm]
            self.print_log("s_shuf", s_shuf)
            optim_chunk_num = int(np.ceil(b_z / 16))
            print "optim_chunk_num::::{}".format(optim_chunk_num)
            v_target_shuf, A_sa_shuf, s_shuf, a_shuf, log_pi_old_sa_shuf = torch.chunk(v_target_shuf, optim_chunk_num), \
                                                                           torch.chunk(A_sa_shuf, optim_chunk_num), \
                                                                           torch.chunk(s_shuf, optim_chunk_num), \
                                                                           torch.chunk(a_shuf, optim_chunk_num), \
                                                                           torch.chunk(log_pi_old_sa_shuf,
                                                                                       optim_chunk_num)
            with open("/home/x/P/DDQ/mylog/ppo/s_shuf_mini.txt", "w") as file:
                file.write(str(s_shuf))
            print "s_shuf_mini.type::::{}".format(type(s_shuf))

            policy_loss, value_loss = 0., 0.
            for v_target_b, A_sa_b, s_b, a_b, log_pi_old_sa_b in zip(v_target_shuf, A_sa_shuf, s_shuf, a_shuf,
                                                                     log_pi_old_sa_shuf):
                self.value_optim.zero_grad()
                v_b = self.ppo_value(s_b).squeeze(-1)
                loss = (v_b - v_target_b).pow(2).mean()
                value_loss += loss.item()
                loss.backward()
                self.value_optim.step()
                self.policy_optim.zero_grad()
                log_pi_sa = self.policy.get_log_prob(s_b, a_b)
                ratio = (log_pi_sa - log_pi_old_sa_b).exp().squeeze(-1)
                ratio = torch.clamp(ratio, 0, 10)
                surrogate1 = ratio * A_sa_b
                surrogate2 = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon) * A_sa_b
                surrogate = - torch.min(surrogate1, surrogate2).mean()
                policy_loss += surrogate.item()
                surrogate.backward()
                for p in self.policy.parameters():
                    p.grad[p.grad != p.grad] = 0.0
                torch.nn.utils.clip_grad_norm(self.policy.parameters(), 10)
                self.policy_optim.step()
            value_loss /= optim_chunk_num
            policy_loss /= optim_chunk_num
            logging.debug('<<dialog policy ppo>> epoch {}, iteration {}, value, loss {}'.format(epoch, i, value_loss))
            logging.debug('<<dialog policy ppo>> epoch {}, iteration {}, policy, loss {}'.format(epoch, i, policy_loss))

        if (epoch + 1) % 10 == 0:
            self.save_ppo("/home/x/P/DDQ/ppo_model", epoch)


        # print "test"
        # sys.exit()

    def save_ppo(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.ppo_value.state_dict(), directory + '/' + str(epoch) + '_ppo.val.mdl')
        torch.save(self.policy.state_dict(), directory + '/' + str(epoch) + '_ppo.pol.mdl')

        logging.info('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))

    ################################################################################
    #    Debug Functions
    ################################################################################
    def save_experience_replay_to_file(self, path):
        """ Save the experience replay pool to a file """

        try:
            pickle.dump(self.experience_replay_pool, open(path, "wb"))
            print 'saved model in %s' % (path,)
        except Exception, e:
            print 'Error: Writing model fails: %s' % (path,)
            print e

    def load_experience_replay_from_file(self, path):
        """ Load the experience replay pool from a file"""

        self.experience_replay_pool = pickle.load(open(path, 'rb'))

    def load_trained_PPO(self, path):
        """ Load the trained DQN from a file """

        trained_file = pickle.load(open(path, 'rb'))
        model = trained_file['model']
        print "Trained PPO Parameters:", json.dumps(trained_file['params'], indent=2)
        return model

    def set_user_planning(self, user_planning):
        self.user_planning = user_planning

    def save_value(self, filename):
        torch.save(self.ppo_value.state_dict(), filename)

    def save_policy(self, filename):
        torch.save(self.policy.state_dict(), filename)

    def load_value(self, filename):
        self.ppo_value.load_state_dict(torch.load(filename))

    def load_policy(self, filename):
        self.policy.load_state_dict(torch.load(filename))


