# -*- coding: utf-8 -*-

from asyncio import Future
import asyncio
from asyncio.queues import Queue
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

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
# from policy_value_network_gpus_tf2 import *
import scipy.stats
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from chess_engine import *

def distrubuted_mcts(game_num, iteration, transfer_data, args, i):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable cuda
    train_mcts = cchess_main(args.train_playout, args.batch_size, True, args.search_threads, 'cpu', args.num_gpus, args.res_block_nums, args.human_color, args.train_epoch, game_num, iteration, transfer_data)
    print("mcts process created and ready to start, process index :" , i+1)
    train_mcts.mcts_process()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'play', 'distributed_train', 'eval'], type=str, help='train or play')
    parser.add_argument('--ai_count', default=1, choices=[0, 1, 2], type=int, help='choose ai player count')
    parser.add_argument('--ai_function', default='mcts', choices=['mcts', 'net'], type=str, help='mcts or net')
    parser.add_argument('--train_playout', default=200, type=int, help='mcts train playout')
    parser.add_argument('--batch_size', default=256, type=int, help='train batch_size')
    parser.add_argument('--play_playout', default=50, type=int, help='mcts play playout')
    parser.add_argument('--delay', dest='delay', action='store',
                        nargs='?', default=1, type=float, required=False,
                        help='Set how many seconds you want to delay after each move')
    parser.add_argument('--end_delay', dest='end_delay', action='store',
                        nargs='?', default=1, type=float, required=False,
                        help='Set how many seconds you want to delay after the end of game')
    parser.add_argument('--search_threads', default=16, type=int, help='search_threads')
    parser.add_argument('--processor', default='gpu', choices=['cpu', 'gpu'], type=str, help='cpu or gpu')
    parser.add_argument('--num_gpus', default=1, type=int, help='gpu counts')
    parser.add_argument('--res_block_nums', default=7, type=int, help='number of res blocks')
    parser.add_argument('--human_color', default='w', choices=['w', 'b'], type=str, help='w (red) or b (black)')
    parser.add_argument('--train_epoch', default=99, type=int, help='number of epochs while training net')
    parser.add_argument('--mcts_num', default=1, type=int, help='number of mcts process in distributed_train mode')
    parser.add_argument('--eval_num', default=1, type=int, help='number of checkpoint from which evaluation start')
    parser.add_argument('--game_num', default=10, type=int, help='number of games in each evaluation')
    args = parser.parse_args()

    if args.mode == 'train':
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        train_main = cchess_main(args.train_playout, args.batch_size, True, args.search_threads, args.processor, args.num_gpus, args.res_block_nums, args.human_color, args.train_epoch)    # * args.num_gpus
        train_main.run()

    elif args.mode == 'play':
        game = Chessgame(args.ai_count, args.ai_function, args.play_playout, args.delay, args.end_delay, args.batch_size,
                         args.search_threads, args.processor, args.num_gpus, args.res_block_nums, args.human_color)    # * args.num_gpus
        game.start()

    elif args.mode == 'distributed_train':
        print("mode : distributed_train")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        mp.set_start_method('spawn')

        # Interprocess communication variable
        game_num = mp.Value("i", 0)
        iteration = mp.Value("i", 0)
        transfer_data = mp.Manager().Queue() 
        
        # create mcts processes
        mcts_num = args.mcts_num
        mcts_sum = []
        print("mcts process number : ", mcts_num, '\n')
        for i in range (0, mcts_num):
            process_name = "mcts" + str(i+1)
            mcts_p = mp.Process(target=distrubuted_mcts, args=(game_num, iteration, transfer_data, args, i,), name=process_name)
            mcts_p.start()
            mcts_sum.append(mcts_p)

        # perform train function in main process
        train_main = cchess_main(args.train_playout, args.batch_size, True, args.search_threads, args.processor, args.num_gpus, args.res_block_nums, args.human_color, args.train_epoch, game_num, iteration, transfer_data)    # * args.num_gpus
        train_main.train_process()
        for i in range (0, mcts_num):
            mcts_sum[i].join()

        print("Training Finished")

    elif args.mode == 'eval':

        checkpoint_dir = "./gpu_models/checkpoints"
        if not os.path.exists(checkpoint_dir):
            print("checkpoint_dir not exist")
            exit()
        latest_ckpt = int(tf.train.latest_checkpoint(checkpoint_dir).split('-')[1]) # get lastest checkpoint number
        # print(tf.train.latest_checkpoint(checkpoint_dir))
        print("latest_ckpt : ", latest_ckpt)

        eval_log_file = open(os.path.join(os.getcwd(), 'eval_log_file.txt'), 'w')

        ai_count = 2
        human_color = 'w'
        args.delay = 0 # delay between each step in visualization
        game = Chessgame(ai_count, args.ai_function, args.play_playout, args.delay, args.end_delay, args.batch_size,
                         args.search_threads, args.processor, args.num_gpus, args.res_block_nums, human_color, if_eval=True)

        # evaluate from checkpoint-{eval_num} and check-point-{eval_num+1}, each evaluation includes {game_num} games
        eval_num = args.eval_num
        game_num = args.game_num
        while (eval_num < latest_ckpt):
            w = b = t = 0 
            print("ready to eval")
            
            for i in range (0, game_num):
                print("game_num : ", i+1, '\n')
                winner = game.eval_start(eval_num)

                if winner == 'error':
                    print("error occured\n")
                    eval_log_file.write("error occured\n")
                    eval_log_file.flush()
                else:
                    if winner == 'w':
                        w = w + 1
                    elif winner == 'b':
                        b = b + 1
                    else:
                        t = t + 1
                
            # exit()

            eval_log_file.write("{} vs {}, {} : {}, tie {} in {} games\n".format(eval_num, eval_num+1, w, b, t, game_num))
            eval_log_file.flush()
            latest_ckpt = int(tf.train.latest_checkpoint(checkpoint_dir).split('-')[1])
            eval_num = eval_num + 1

        eval_log_file.close()
        print("Eval Finished")