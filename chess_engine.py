# -*- coding: utf-8 -*-

from pydoc import doc
import tensorflow as tf
import numpy as np
import os
import sys
import random
import time
import argparse
from collections import deque, defaultdict, namedtuple
import copy
from policy_value_net import *
from policy_value_net_cpu import *
from chess_game import *
from chess_board import *
import scipy.stats
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from board_utils import *
from mcts import *
from params import *
from chess_utils import raise_error

import multiprocessing as mp
from multiprocessing.managers import BaseManager

def getManager():
    m = BaseManager()
    m.start()
    return m

def softmax(x):
    # print(x)
    probs = np.exp(x - np.max(x))
    # print(np.sum(probs))
    probs /= np.sum(probs)
    return probs

class cchess_main(object):

    def __init__(self, playout=400, in_batch_size=128, exploration=True, in_search_threads=16, processor="cpu", num_gpus=1, res_block_nums=7, human_color='b', train_epoch=42, game_num=0, iteration=0, transfer_data=0):
        self.epochs = 5
        self.train_epoch = train_epoch
        self.playout_counts = playout    #400    #800    #1600    200
        self.temperature = 1    #1e-8    1e-3
        # self.c = 1e-4
        self.batch_size = in_batch_size    #128    #512
        # self.momentum = 0.9
        self.game_batch = 400    #  Evaluation each 400 times
        # self.game_loop = 25000
        self.top_steps = 30
        self.top_temperature = 1    #2
        # self.Dirichlet = 0.3    # P(s,a) = (1 - ϵ)p_a  + ϵη_a    #self-play chapter in the paper
        self.eta = 0.03
        # self.epsilon = 0.25
        # self.v_resign = 0.05
        # self.c_puct = 5
        self.learning_rate = 0.001    #5e-3    #    0.001
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.buffer_size = 10000
        self.data_buffer = deque(maxlen=self.buffer_size)        
        self.game_borad =  Boardutils(human_color=human_color)
        self.processor = processor
        # self.current_player = 'w'    #“w”表示红方，“b”表示黑方。
        self.policy_value_netowrk = policy_value_net_cpu(self.lr_callback, res_block_nums) if processor == 'cpu' else policy_value_net(num_gpus, res_block_nums)
        self.search_threads = in_search_threads
        self.mcts = MCTS_tree(self.game_borad.state, self.policy_value_netowrk.forward, self.search_threads)
        self.exploration = exploration
        self.resign_threshold = -0.8    #0.05
        self.global_step = 0
        self.kl_targ = 0.025
        self.log_file = open(os.path.join(os.getcwd(), 'log_file.txt'), 'w')
        self.human_color = human_color
        self.game_num = game_num
        self.iteration = iteration
        self.transfer_data = transfer_data

    @staticmethod
    def flip_policy(prob):
        prob = tf.squeeze(prob) # .flatten()
        return np.asarray([prob[ind] for ind in unflipped_index])

    def lr_callback(self):
        return self.learning_rate * self.lr_multiplier

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        #print("training data_buffer len : ", len(self.data_buffer))
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        winner_batch = np.expand_dims(winner_batch, 1)

        start_time = time.time()
        old_probs, old_v = self.mcts.forward(state_batch)
        for i in range(self.epochs):
            # print("tf.executing_eagerly() : ", tf.executing_eagerly())
            state_batch = np.array(state_batch)
            if len(state_batch.shape) == 3:
                sp = state_batch.shape
                state_batch = np.reshape(state_batch, [1, sp[0], sp[1], sp[2]])
            if self.processor == 'cpu':
                accuracy, loss, self.global_step = self.policy_value_netowrk.train_step(state_batch, mcts_probs_batch, winner_batch,
                                                                 self.learning_rate * self.lr_multiplier)    #
            else:
                # import pickle
                # pickle.dump((state_batch, mcts_probs_batch, winner_batch, self.learning_rate * self.lr_multiplier), open('preprocess.p', 'wb'))
                with self.policy_value_netowrk.strategy.scope():
                    train_dataset = tf.data.Dataset.from_tensor_slices((state_batch, mcts_probs_batch, winner_batch)).batch(len(winner_batch))  # , self.learning_rate * self.lr_multiplier
                        # .shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
                    train_iterator = self.policy_value_netowrk.strategy.make_dataset_iterator(train_dataset)
                    train_iterator.initialize()
                    accuracy, loss, self.global_step = self.policy_value_netowrk.distributed_train(train_iterator)

            new_probs, new_v = self.mcts.forward(state_batch)
            old_probs = np.where(old_probs > 1.0e-10, old_probs, 1.0e-10)
            new_probs = np.where(new_probs > 1.0e-10, new_probs, 1.0e-10)
            kl_tmp = old_probs * (np.log((old_probs + 1e-10) / (new_probs + 1e-10)))

            kl_lst = []
            for line in kl_tmp:
                # print("line.shape", line.shape)
                all_value = [x for x in line if str(x) != 'nan' and str(x)!= 'inf']#除去inf值
                kl_lst.append(np.sum(all_value))
            kl = np.mean(kl_lst)
            # kl = scipy.stats.entropy(old_probs, new_probs)
            # kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))

            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # self.policy_value_netowrk.save()
        print("[Policy-Value-Net] -> Training Took {} s".format(time.time() - start_time))

        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = 1 - np.var(np.array(winner_batch) - tf.squeeze(old_v)) / np.var(np.array(winner_batch)) # .flatten()
        explained_var_new = 1 - np.var(np.array(winner_batch) - tf.squeeze(new_v)) / np.var(np.array(winner_batch)) # .flatten()
        print(
            "[Policy-Value-Net] -> KL Divergence:{}; \t\tLR Multiplier:{}; \n[Policy-Value-Net] -> Loss:{}; \t\t\tAccuracy:{}; \n[Policy-Value-Net] -> Explained Var (old):{}; \t\t\tExplained Var (new):{};\n".format(
                kl, self.lr_multiplier, loss, accuracy, explained_var_old, explained_var_new))
        self.log_file.write("[Policy-Value-Net] -> KL Divergence:{}; \t\tLR Multiplier:{}; \n[Policy-Value-Net] -> Loss:{}; \t\t\tAccuracy:{}; \n[Policy-Value-Net] -> Explained Var (old):{}; \t\t\tExplained Var (new):{};".format(
                kl, self.lr_multiplier, loss, accuracy, explained_var_old, explained_var_new) + '\n')
        self.log_file.flush()
        # return loss, accuracy

    def mcts_process(self):
        # self.processor = "cpu"
        print("mcts process entered")
        while(self.iteration.value <= self.train_epoch):
            # start_time = time.time()
            self.policy_value_netowrk.checkpoint.restore(tf.train.latest_checkpoint(self.policy_value_netowrk.checkpoint_dir))
            # print("**************************************************")
            # print("Restore Took {} s".format(time.time() - start_time))
            # print("**************************************************")

            play_data, episode_len = self.selfplay()
            extend_data = []
            # states_data = []
            for state, mcts_prob, winner in play_data:
                states_data = self.mcts.state_to_positions(state)
                extend_data.append((states_data, mcts_prob, winner))

            self.transfer_data.put(extend_data)
            # print("data pushed\n")
            self.game_num.value += 1

        print("mcts process exited")

    def train_process(self):
        # self.game_loop
        start_time = time.time()
        print("[Train CChess] -> Training Start ({} Epochs)\n".format(self.train_epoch))

        try:
            # time.sleep(10)
            self.iteration.value = 0
            total_data_len = 0
            while(self.iteration.value <= self.train_epoch):
                # print("**************")
                print("game_num : ", self.game_num.value)
                # print("**************")
                # print("self.transfer_data.empty: ", self.transfer_data.empty())

                if self.transfer_data.empty() == False:
                    while (self.transfer_data.empty() == False):
                        extend_data = self.transfer_data.get()
                        self.data_buffer.extend(extend_data)
                        total_data_len = total_data_len + len(extend_data)
                        print(".")  
                    print("data pulled")
                self.log_file.write("time:{}\t\ttotal_data_len:{}".format(time.time()-start_time, total_data_len) + '\n')
                self.log_file.flush()  

                print("training data_buffer len : ", len(self.data_buffer))
                if len(self.data_buffer) > self.batch_size:
                    self.iteration.value = self.iteration.value + 1
                    print("**************")
                    print("iteration: ", self.iteration.value)
                    print("**************")
                    self.log_file.write("iteration:{}".format(self.iteration.value) + '\n')
                    self.log_file.write("game_num:{}".format(self.game_num.value) + '\n')
                    self.log_file.flush()
                    
                    self.policy_update()
                    if self.iteration.value % 10 == 0:
                        self.policy_value_netowrk.save()
                        print("Network saved\n")
                else:
                    time.sleep(30)

            self.log_file.close()
            
            self.policy_value_netowrk.save()

            print("[Train CChess] -> Training Finished, Took {}s".format(time.time() - start_time))

        except KeyboardInterrupt:
            raise_error(__file__, sys._getframe().f_lineno, \
                message=("CChess Training Finished (KeyboardInterrupt) and Model Saved"), color="yellow")
            self.log_file.close()
            self.policy_value_netowrk.save()

    def run(self):
        #self.game_loop
        batch_iter = 0
        start_time = time.time()
        print("[Train CChess] -> Training Start ({} Epochs)".format(self.train_epoch ))

        try:
            total_data_len = 0
            while(batch_iter <= self.train_epoch):
                batch_iter += 1
                play_data, episode_len = self.selfplay()
                print("[Train CChess] -> Batch {}/{}; Episode Length:{}; Iteration:{}".format(batch_iter, self.train_epoch, episode_len, batch_iter))
                extend_data = []
                # states_data = []
                for state, mcts_prob, winner in play_data:
                    states_data = self.mcts.state_to_positions(state)
                    extend_data.append((states_data, mcts_prob, winner))
                self.data_buffer.extend(extend_data)
                total_data_len = total_data_len + len(extend_data)
                self.log_file.write("time:{}\t\ttotal_data_len:{}".format(time.time()-start_time, total_data_len) + '\n')
                self.log_file.flush()
                print("training data_buffer len : ", len(self.data_buffer))  
                if len(self.data_buffer) > self.batch_size:
                    self.policy_update()

            self.log_file.close()
            self.policy_value_netowrk.save()
            print("[Train CChess] -> Training Finished, Took {}s".format(time.time() - start_time))

        except KeyboardInterrupt:
            raise_error(__file__, sys._getframe().f_lineno, \
                message=("CChess Training Finished (KeyboardInterrupt) and Model Saved"), color="yellow")
            self.log_file.close()
            self.policy_value_netowrk.save()

    def get_hint(self, mcts_or_net, reverse, disp_mcts_msg_handler):

        if mcts_or_net == "mcts":
            disp_mcts_msg_handler()
            if self.mcts.root.child == {}:
                self.mcts.main(self.game_borad.state, self.game_borad.current_player, self.game_borad.restrict_round,
                               self.playout_counts)

            actions_visits = [(act, nod.N) for act, nod in self.mcts.root.child.items()]
            actions, visits = zip(*actions_visits)
            # print("visits : ", visits)
            # print("np.log(visits) : ", np.log(visits))
            probs = softmax(1.0 / self.temperature * np.log(visits))  # + 1e-10

            act_prob_dict = defaultdict(float)
            for i in range(len(actions)):
                if self.human_color == 'w':
                    action = "".join(flipped_uci_labels(actions[i]))
                else:
                    action = actions[i]
                act_prob_dict[action] = probs[i]

        elif mcts_or_net == "net":
            positions = self.mcts.generate_inputs(self.game_borad.state, self.game_borad.current_player)
            positions = np.expand_dims(positions, 0)
            action_probs, value = self.mcts.forward(positions)

            if self.mcts.is_black_turn(self.game_borad.current_player):
                action_probs = cchess_main.flip_policy(action_probs)
            moves =  Boardutils.get_legal_moves(self.game_borad.state, self.game_borad.current_player)

            tot_p = 1e-8
            action_probs = tf.squeeze(action_probs)  # .flatten()  # .squeeze()
            act_prob_dict = defaultdict(float)
            # print("expand action_probs shape : ", action_probs.shape)
            for action in moves:
                # in_state =  Boardutils.sim_do_action(action, self.state)
                mov_p = action_probs[label2i[action]]
                if self.human_color == 'w':
                    action = "".join(flipped_uci_labels(action))
                act_prob_dict[action] = mov_p
                # new_node = leaf_node(self, mov_p, in_state)
                # self.child[action] = new_node
                tot_p += mov_p

            for a, _ in act_prob_dict.items():
                act_prob_dict[a] /= tot_p

        sorted_move_probs = sorted(act_prob_dict.items(), key=lambda item: item[1], reverse=reverse)
        # print(sorted_move_probs)

        return sorted_move_probs

    #@profile
    def get_action(self, state, temperature = 1e-3):
        # for i in range(self.playout_counts):
        #     state_sim = copy.deepcopy(state)
        #     self.mcts.do_simulation(state_sim, self.game_borad.current_player, self.game_borad.restrict_round)
        
        self.mcts.main(state, self.game_borad.current_player, self.game_borad.restrict_round, self.playout_counts)

        actions_visits = [(act, nod.N) for act, nod in self.mcts.root.child.items()]
        actions, visits = zip(*actions_visits)
        # print(visits)
        # print(type(visits))
        visits_tmp = np.array(list(visits))
        # print(visits_tmp)
        # print(type(visits_tmp))
        visits_tmp = np.where(visits_tmp > 1.0e-10, visits_tmp, 1.0e-10)
        # print(visits_tmp)
        # print(type(visits_tmp))
        probs = softmax(1.0 / temperature * np.log(visits_tmp, where=visits_tmp > 0))    #+ 1e-10
        # print(probs)
        # print(type(probs))
        # probs = tuple(list(probs))
        # print(probs)
        # print(type(probs))
        # exit()
        move_probs = []
        move_probs.append([actions, probs])

        if(self.exploration):
            act = np.random.choice(actions, p=0.75 * probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
        else:
            act = np.random.choice(actions, p=probs)

        win_rate = self.mcts.Q(act) # / 2.0 + 0.5
        self.mcts.update_tree(act)

        return act, move_probs, win_rate

    def check_end(self):
        if (self.game_borad.state.find('K') == -1 or self.game_borad.state.find('k') == -1):
            if (self.game_borad.state.find('K') == -1):
                print("Black is Winner")
                return True, "b"
            if (self.game_borad.state.find('k') == -1):
                print("Red is Winner")
                return True, "w"
        elif self.game_borad.restrict_round >= tie_tolerance:
            print("TIE! No Winners!")
            return True, "t"
        else:
            return False, ""

    def human_move(self, coord, mcts_or_net):
        win_rate = 0
        x_trans = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i'}

        src = coord[0:2]
        dst = coord[2:4]

        src_x = (x_trans[src[0]])
        src_y = str(src[1])

        dst_x = (x_trans[dst[0]])
        dst_y = str(dst[1])

        action = src_x + src_y + dst_x + dst_y

        if self.human_color == 'w':
            action = "".join(flipped_uci_labels(action))

        if mcts_or_net == "mcts":
            if self.mcts.root.child == {}:
                # self.get_action(self.game_borad.state, self.temperature)
                self.mcts.main(self.game_borad.state, self.game_borad.current_player, self.game_borad.restrict_round,
                               self.playout_counts)
            win_rate = self.mcts.Q(action) # / 2.0 + 0.5
            self.mcts.update_tree(action)

        last_state = self.game_borad.state
        # print(self.game_borad.current_player, " now take a action : ", action, "[Step {}]".format(self.game_borad.round))
        self.game_borad.state =  Boardutils.sim_do_action(action, self.game_borad.state)
        self.game_borad.round += 1
        self.game_borad.current_player = "w" if self.game_borad.current_player == "b" else "b"
        if is_kill_move(last_state, self.game_borad.state) == 0:
            self.game_borad.restrict_round += 1
        else:
            self.game_borad.restrict_round = 0

        return win_rate


    def select_move(self, mcts_or_net):
        if mcts_or_net == "mcts":
            action, probs, win_rate = self.get_action(self.game_borad.state, self.temperature)
            # win_rate = self.mcts.Q(action) / 2.0 + 0.5
        elif mcts_or_net == "net":
            positions = self.mcts.generate_inputs(self.game_borad.state, self.game_borad.current_player)
            positions = np.expand_dims(positions, 0)
            action_probs, value = self.mcts.forward(positions)
            win_rate = value[0, 0] # / 2 + 0.5
            if self.mcts.is_black_turn(self.game_borad.current_player):
                action_probs = cchess_main.flip_policy(action_probs)
            moves =  Boardutils.get_legal_moves(self.game_borad.state, self.game_borad.current_player)

            tot_p = 1e-8
            action_probs = tf.squeeze(action_probs)  # .flatten()  # .squeeze()
            act_prob_dict = defaultdict(float)
            # print("expand action_probs shape : ", action_probs.shape)
            for action in moves:
                # in_state =  Boardutils.sim_do_action(action, self.state)
                mov_p = action_probs[label2i[action]]
                act_prob_dict[action] = mov_p
                # new_node = leaf_node(self, mov_p, in_state)
                # self.child[action] = new_node
                tot_p += mov_p

            for a, _ in act_prob_dict.items():
                act_prob_dict[a] /= tot_p

            action = max(act_prob_dict.items(), key=lambda node: node[1])[0]
            # self.mcts.update_tree(action)

        print('Win Rate for Player {} (AI) is {:.4f}'.format("Black" if self.game_borad.current_player=="b" else "Red", win_rate))
        last_state = self.game_borad.state
        print("Black" if self.game_borad.current_player=="b" else "Red", "(AI) Took Action ->", action, "[Step {}]".format(self.game_borad.round))    # if self.human_color == 'w' else "".join(flipped_uci_labels(action))
        self.game_borad.state =  Boardutils.sim_do_action(action, self.game_borad.state)
        self.game_borad.round += 1
        self.game_borad.current_player = "w" if self.game_borad.current_player == "b" else "b"
        if is_kill_move(last_state, self.game_borad.state) == 0:
            self.game_borad.restrict_round += 1
        else:
            self.game_borad.restrict_round = 0

        self.game_borad.print_borad(self.game_borad.state)

        x_trans = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}

        if self.human_color == 'w':
            action = "".join(flipped_uci_labels(action))

        src = action[0:2]
        dst = action[2:4]

        src_x = int(x_trans[src[0]])
        src_y = int(src[1])

        dst_x = int(x_trans[dst[0]])
        dst_y = int(dst[1])

        return (src_x, src_y, dst_x - src_x, dst_y - src_y), win_rate

    def selfplay(self):
        print("[Self-Play] -> Begin")
        self.game_borad.reload()
        # p1, p2 = self.game_borad.players
        states, mcts_probs, current_players = [], [], []
        z = None
        game_over = False
        winnner = ""
        start_time = time.time()
        # self.game_borad.print_borad(self.game_borad.state)
        while(not game_over):
            # print("ready to get action")
            action, probs, win_rate = self.get_action(self.game_borad.state, self.temperature)
            # print("action got")
            state, palyer = self.mcts.try_flip(self.game_borad.state, self.game_borad.current_player, self.mcts.is_black_turn(self.game_borad.current_player))
            states.append(state)
            prob = np.zeros(labels_len)
            if self.mcts.is_black_turn(self.game_borad.current_player):
                for idx in range(len(probs[0][0])):
                    # probs[0][0][idx] = "".join((str(9 - int(a)) if a.isdigit() else a) for a in probs[0][0][idx])
                    act = "".join((str(9 - int(a)) if a.isdigit() else a) for a in probs[0][0][idx])
                    # for idx in range(len(mcts_prob[0][0])):
                    prob[label2i[act]] = probs[0][1][idx]
            else:
                for idx in range(len(probs[0][0])):
                    prob[label2i[probs[0][0][idx]]] = probs[0][1][idx]
            mcts_probs.append(prob)
            # mcts_probs.append(probs)
            current_players.append(self.game_borad.current_player)

            last_state = self.game_borad.state
            # print(self.game_borad.current_player, " now take a action : ", action, "[Step {}]".format(self.game_borad.round))
            self.game_borad.state =  Boardutils.sim_do_action(action, self.game_borad.state)
            self.game_borad.round += 1
            self.game_borad.current_player = "w" if self.game_borad.current_player == "b" else "b"
            if is_kill_move(last_state, self.game_borad.state) == 0:
                self.game_borad.restrict_round += 1
            else:
                self.game_borad.restrict_round = 0

            # self.game_borad.print_borad(self.game_borad.state, action)

            if (self.game_borad.state.find('K') == -1 or self.game_borad.state.find('k') == -1):
                z = np.zeros(len(current_players))
                if (self.game_borad.state.find('K') == -1):
                    winnner = "b"
                if (self.game_borad.state.find('k') == -1):
                    winnner = "w"
                z[np.array(current_players) == winnner] = 1.0
                z[np.array(current_players) != winnner] = -1.0
                game_over = True
                print("[Self-Play] -> Game End. Winner is Player ->", ("Black" if winnner=="b" else "Red"), " In {} steps".format(self.game_borad.round - 1))
            elif self.game_borad.restrict_round >= tie_tolerance_train:
                z = np.zeros(len(current_players))
                game_over = True
                print("[Self-Play] -> Game End. Tie in {} Steps".format(self.game_borad.round - 1))
            # elif(self.mcts.root.v < self.resign_threshold):
            #     pass
            # elif(self.mcts.root.Q < self.resign_threshold):
            #    pass
            if(game_over):
                # self.mcts.root = leaf_node(None, self.mcts.p_, "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr")#"rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"
                self.mcts.reload()
        print("[Self-Play] -> Took {} s\n".format(time.time() - start_time))
        return zip(states, mcts_probs, z), len(z)
