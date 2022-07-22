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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'play'], type=str, help='train or play')
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
    args = parser.parse_args()

    if args.mode == 'train':
        train_main = cchess_main(args.train_playout, args.batch_size, True, args.search_threads, args.processor, args.num_gpus, args.res_block_nums, args.human_color, args.train_epoch)    # * args.num_gpus
        train_main.run()
    elif args.mode == 'play':
        game = Chessgame(args.ai_count, args.ai_function, args.play_playout, args.delay, args.end_delay, args.batch_size,
                         args.search_threads, args.processor, args.num_gpus, args.res_block_nums, args.human_color)    # * args.num_gpus
        game.start()
