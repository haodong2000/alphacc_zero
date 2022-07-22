# -*- coding: utf-8 -*-

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


def flipped_uci_labels(param):
    def repl(x):
        return "".join([(str(9 - int(a)) if a.isdigit() else a) for a in x])

    return [repl(x) for x in param]

# 创建所有合法走子UCI，size 2086
def create_uci_labels():
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    Advisor_labels = ['d7e8', 'e8d7', 'e8f9', 'f9e8', 'd0e1', 'e1d0', 'e1f2', 'f2e1',
                      'd2e1', 'e1d2', 'e1f0', 'f0e1', 'd9e8', 'e8d9', 'e8f7', 'f7e8']
    Bishop_labels = ['a2c4', 'c4a2', 'c0e2', 'e2c0', 'e2g4', 'g4e2', 'g0i2', 'i2g0',
                     'a7c9', 'c9a7', 'c5e7', 'e7c5', 'e7g9', 'g9e7', 'g5i7', 'i7g5',
                     'a2c0', 'c0a2', 'c4e2', 'e2c4', 'e2g0', 'g0e2', 'g4i2', 'i2g4',
                     'a7c5', 'c5a7', 'c9e7', 'e7c9', 'e7g5', 'g5e7', 'g9i7', 'i7g9']
    # King_labels = ['d0d7', 'd0d8', 'd0d9', 'd1d7', 'd1d8', 'd1d9', 'd2d7', 'd2d8', 'd2d9',
    #                'd7d0', 'd7d1', 'd7d2', 'd8d0', 'd8d1', 'd8d2', 'd9d0', 'd9d1', 'd9d2',
    #                'd0d7', 'd0d8', 'd0d9', 'd1d7', 'd1d8', 'd1d9', 'd2d7', 'd2d8', 'd2d9',
    #                'd0d7', 'd0d8', 'd0d9', 'd1d7', 'd1d8', 'd1d9', 'd2d7', 'd2d8', 'd2d9',
    #                'd0d7', 'd0d8', 'd0d9', 'd1d7', 'd1d8', 'd1d9', 'd2d7', 'd2d8', 'd2d9',
    #                'd0d7', 'd0d8', 'd0d9', 'd1d7', 'd1d8', 'd1d9', 'd2d7', 'd2d8', 'd2d9']

    for l1 in range(9):
        for n1 in range(10):
            destinations = [(t, n1) for t in range(9)] + \
                           [(l1, t) for t in range(10)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]  # 马走日
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(9) and n2 in range(10):
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                    labels_array.append(move)

    for p in Advisor_labels:
        labels_array.append(p)

    for p in Bishop_labels:
        labels_array.append(p)

    return labels_array

def create_position_labels():
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    letters.reverse()
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for l1 in range(9):
        for n1 in range(10):
            move = letters[8 - l1] + numbers[n1]
            labels_array.append(move)
#     labels_array.reverse()
    return labels_array

def create_position_labels_reverse():
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    letters.reverse()
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for l1 in range(9):
        for n1 in range(10):
            move = letters[l1] + numbers[n1]
            labels_array.append(move)
    labels_array.reverse()
    return labels_array

class Boardutils(object):
    board_pos_name = np.array(create_position_labels()).reshape(9,10).transpose()
    Ny = 10
    Nx = 9
    north_is_red = False

    def __init__(self, human_color):
        self.state = "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr"#"rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"    #
        self.round = 1
        # self.players = ["w", "b"]
        self.current_player = "w"
        self.restrict_round = 0
        Boardutils.north_is_red = (human_color == 'b')

# 小写表示黑方，大写表示红方
# [
#     "rheakaehr",
#     "         ",
#     " c     c ",
#     "p p p p p",
#     "         ",
#     "         ",
#     "P P P P P",
#     " C     C ",
#     "         ",
#     "RHEAKAEHR"
# ]
    def reload(self):
        self.state = "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr"#"rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"    #
        self.round = 1
        self.current_player = "w"
        self.restrict_round = 0

    @staticmethod
    def print_borad(board, action = None):
        def string_reverse(string):
            # return ''.join(string[len(string) - i] for i in range(1, len(string)+1))
            return ''.join(string[i] for i in range(len(string) - 1, -1, -1))

        x_trans = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}

        if(action != None):
            src = action[0:2]

            src_x = int(x_trans[src[0]])
            src_y = int(src[1])

        # board = string_reverse(board)
        board = board.replace("1", " ")
        board = board.replace("2", "  ")
        board = board.replace("3", "   ")
        board = board.replace("4", "    ")
        board = board.replace("5", "     ")
        board = board.replace("6", "      ")
        board = board.replace("7", "       ")
        board = board.replace("8", "        ")
        board = board.replace("9", "         ")
        board = board.split('/')
        # board = board.replace("/", "\n")
        if Boardutils.north_is_red:
            print("\n  \tA\tB\tC\tD\tE\tF\tG\tH\tI\n")
            for i, line in enumerate(board):
                if (action != None):
                    if(i == src_y):
                        s = list(line)
                        s[src_x] = 'x'
                        line = ''.join(s)
                new_line = ""
                for char in line:
                    new_line += chess_en[char] + "\t"
                print(i, "\t" + new_line, end="\n\n")
            print("-*" * 40 + "-\n")
            print("  \tA\tB\tC\tD\tE\tF\tG\tH\tI\n")
            for i, line in enumerate(board):
                if (action != None):
                    if(i == src_y):
                        s = list(line)
                        s[src_x] = 'x'
                        line = ''.join(s)
                new_line = ""
                for char in line:
                    new_line += chess_cn[char] + "\t"
                print(i, "\t" + new_line, end="\n\n")
        else:
            print("  \tA\tB\tC\tD\tE\tF\tG\tH\tI\n")
            for i, line in reversed(list(enumerate(board))):
                if (action != None):
                    if(i == src_y):
                        s = list(line)
                        s[src_x] = 'x'
                        line = ''.join(s)
                new_line = ""
                for char in line:
                    new_line += chess_en[char] + "\t"
                print(i, "\t" + new_line, end="\n\n")
            print("-*" * 40 + "-\n")
            print("  \tA\tB\tC\tD\tE\tF\tG\tH\tI\n")
            for i, line in reversed(list(enumerate(board))):
                if (action != None):
                    if(i == src_y):
                        s = list(line)
                        s[src_x] = 'x'
                        line = ''.join(s)
                new_line = ""
                for char in line:
                    new_line += chess_cn[char] + "\t"
                print(i, "\t" + new_line, end="\n\n")
        # print(board)

    @staticmethod
    def sim_do_action(in_action, in_state):
        x_trans = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8}

        src = in_action[0:2]
        dst = in_action[2:4]

        src_x = int(x_trans[src[0]])
        src_y = int(src[1])

        dst_x = int(x_trans[dst[0]])
        dst_y = int(dst[1])

        # Boardutils.print_borad(in_state)
        # print("sim_do_action : ", in_action)
        # print(dst_y, dst_x, src_y, src_x)
        board_positions = Boardutils.board_to_pos_name(in_state)
        line_lst = []
        for line in board_positions:
            line_lst.append(list(line))
        lines = np.array(line_lst)
        # print(lines.shape)
        # print(board_positions[src_y])
        # print("before board_positions[dst_y] = ",board_positions[dst_y])

        lines[dst_y][dst_x] = lines[src_y][src_x]
        lines[src_y][src_x] = '1'

        board_positions[dst_y] = ''.join(lines[dst_y])
        board_positions[src_y] = ''.join(lines[src_y])

        # src_str = list(board_positions[src_y])
        # dst_str = list(board_positions[dst_y])
        # print("src_str[src_x] = ", src_str[src_x])
        # print("dst_str[dst_x] = ", dst_str[dst_x])
        # c = copy.deepcopy(src_str[src_x])
        # dst_str[dst_x] = c
        # src_str[src_x] = '1'
        # board_positions[dst_y] = ''.join(dst_str)
        # board_positions[src_y] = ''.join(src_str)
        # print("after board_positions[dst_y] = ", board_positions[dst_y])

        # board_positions[dst_y][dst_x] = board_positions[src_y][src_x]
        # board_positions[src_y][src_x] = '1'

        board = "/".join(board_positions)
        board = board.replace("111111111", "9")
        board = board.replace("11111111", "8")
        board = board.replace("1111111", "7")
        board = board.replace("111111", "6")
        board = board.replace("11111", "5")
        board = board.replace("1111", "4")
        board = board.replace("111", "3")
        board = board.replace("11", "2")

        # Boardutils.print_borad(board)
        return board

    @staticmethod
    def board_to_pos_name(board):
        board = board.replace("2", "11")
        board = board.replace("3", "111")
        board = board.replace("4", "1111")
        board = board.replace("5", "11111")
        board = board.replace("6", "111111")
        board = board.replace("7", "1111111")
        board = board.replace("8", "11111111")
        board = board.replace("9", "111111111")
        return board.split("/")

    @staticmethod
    def check_bounds(toY, toX):
        if toY < 0 or toX < 0:
            return False

        if toY >= Boardutils.Ny or toX >= Boardutils.Nx:
            return False

        return True

    @staticmethod
    def validate_move(c, upper=True):
        if (c.isalpha()):
            if (upper == True):
                if (c.islower()):
                    return True
                else:
                    return False
            else:
                if (c.isupper()):
                    return True
                else:
                    return False
        else:
            return True

    @staticmethod
    def get_legal_moves(state, current_player):
        moves = []
        k_x = None
        k_y = None

        K_x = None
        K_y = None

        face_to_face = False

        board_positions = np.array(Boardutils.board_to_pos_name(state))
        for y in range(board_positions.shape[0]):
            for x in range(len(board_positions[y])):
                if(board_positions[y][x].isalpha()):
                    if(board_positions[y][x] == 'r' and current_player == 'b'):
                        toY = y
                        for toX in range(x - 1, -1, -1):
                            m = Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].isupper()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toX in range(x + 1, Boardutils.Nx):
                            m = Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].isupper()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        toX = x
                        for toY in range(y - 1, -1, -1):
                            m = Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].isupper()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toY in range(y + 1, Boardutils.Ny):
                            m = Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].isupper()):
                                    moves.append(m)
                                break

                            moves.append(m)

                    elif(board_positions[y][x] == 'R' and current_player == 'w'):
                        toY = y
                        for toX in range(x - 1, -1, -1):
                            m = Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].islower()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toX in range(x + 1, Boardutils.Nx):
                            m = Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].islower()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        toX = x
                        for toY in range(y - 1, -1, -1):
                            m = Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].islower()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toY in range(y + 1, Boardutils.Ny):
                            m = Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].islower()):
                                    moves.append(m)
                                break

                            moves.append(m)

                    elif ((board_positions[y][x] == 'n' or board_positions[y][x] == 'h') and current_player == 'b'):
                        for i in range(-1, 3, 2):
                            for j in range(-1, 3, 2):
                                toY = y + 2 * i
                                toX = x + 1 * j
                                if Boardutils.check_bounds(toY, toX) and Boardutils.validate_move(board_positions[toY][toX], upper=False) and board_positions[toY - i][x].isalpha() == False:
                                    moves.append(Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX])
                                toY = y + 1 * i
                                toX = x + 2 * j
                                if Boardutils.check_bounds(toY, toX) and Boardutils.validate_move(board_positions[toY][toX], upper=False) and board_positions[y][toX - j].isalpha() == False:
                                    moves.append(Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX])
                    elif ((board_positions[y][x] == 'N' or board_positions[y][x] == 'H') and current_player == 'w'):
                        for i in range(-1, 3, 2):
                            for j in range(-1, 3, 2):
                                toY = y + 2 * i
                                toX = x + 1 * j
                                if Boardutils.check_bounds(toY, toX) and Boardutils.validate_move(board_positions[toY][toX], upper=True) and board_positions[toY - i][x].isalpha() == False:
                                    moves.append(Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX])
                                toY = y + 1 * i
                                toX = x + 2 * j
                                if Boardutils.check_bounds(toY, toX) and Boardutils.validate_move(board_positions[toY][toX], upper=True) and board_positions[y][toX - j].isalpha() == False:
                                    moves.append(Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX])
                    elif ((board_positions[y][x] == 'b' or board_positions[y][x] == 'e') and current_player == 'b'):
                        for i in range(-2, 3, 4):
                            toY = y + i
                            toX = x + i

                            if Boardutils.check_bounds(toY, toX) and Boardutils.validate_move(board_positions[toY][toX],
                                                                        upper=False) and toY >= 5 and \
                                            board_positions[y + i // 2][x + i // 2].isalpha() == False:
                                moves.append(Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX])
                            toY = y + i
                            toX = x - i

                            if Boardutils.check_bounds(toY, toX) and Boardutils.validate_move(board_positions[toY][toX],
                                                                        upper=False) and toY >= 5 and \
                                            board_positions[y + i // 2][x - i // 2].isalpha() == False:
                                moves.append(Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX])
                    elif ((board_positions[y][x] == 'B' or board_positions[y][x] == 'E') and current_player == 'w'):
                        for i in range(-2, 3, 4):
                            toY = y + i
                            toX = x + i

                            if Boardutils.check_bounds(toY, toX) and Boardutils.validate_move(board_positions[toY][toX],
                                                                        upper=True) and toY <= 4 and \
                                            board_positions[y + i // 2][x + i // 2].isalpha() == False:
                                moves.append(Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX])
                            toY = y + i
                            toX = x - i

                            if Boardutils.check_bounds(toY, toX) and Boardutils.validate_move(board_positions[toY][toX],
                                                                        upper=True) and toY <= 4 and \
                                            board_positions[y + i // 2][x - i // 2].isalpha() == False:
                                moves.append(Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX])
                    elif (board_positions[y][x] == 'a' and current_player == 'b'):
                        for i in range(-1, 3, 2):
                            toY = y + i
                            toX = x + i

                            if Boardutils.check_bounds(toY, toX) and Boardutils.validate_move(board_positions[toY][toX],
                                                                        upper=False) and toY >= 7 and toX >= 3 and toX <= 5:
                                moves.append(Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX])

                            toY = y + i
                            toX = x - i

                            if Boardutils.check_bounds(toY, toX) and Boardutils.validate_move(board_positions[toY][toX],
                                                                        upper=False) and toY >= 7 and toX >= 3 and toX <= 5:
                                moves.append(Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX])
                    elif (board_positions[y][x] == 'A' and current_player == 'w'):
                        for i in range(-1, 3, 2):
                            toY = y + i
                            toX = x + i

                            if Boardutils.check_bounds(toY, toX) and Boardutils.validate_move(board_positions[toY][toX],
                                                                        upper=True) and toY <= 2 and toX >= 3 and toX <= 5:
                                moves.append(Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX])

                            toY = y + i
                            toX = x - i

                            if Boardutils.check_bounds(toY, toX) and Boardutils.validate_move(board_positions[toY][toX],
                                                                        upper=True) and toY <= 2 and toX >= 3 and toX <= 5:
                                moves.append(Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX])
                    elif (board_positions[y][x] == 'k'):
                        k_x = x
                        k_y = y

                        if(current_player == 'b'):
                            for i in range(2):
                                for sign in range(-1, 2, 2):
                                    j = 1 - i
                                    toY = y + i * sign
                                    toX = x + j * sign

                                    if Boardutils.check_bounds(toY, toX) and Boardutils.validate_move(board_positions[toY][toX],
                                                                                upper=False) and toY >= 7 and toX >= 3 and toX <= 5:
                                        moves.append(Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX])
                    elif (board_positions[y][x] == 'K'):
                        K_x = x
                        K_y = y

                        if(current_player == 'w'):
                            for i in range(2):
                                for sign in range(-1, 2, 2):
                                    j = 1 - i
                                    toY = y + i * sign
                                    toX = x + j * sign

                                    if Boardutils.check_bounds(toY, toX) and Boardutils.validate_move(board_positions[toY][toX],
                                                                                upper=True) and toY <= 2 and toX >= 3 and toX <= 5:
                                        moves.append(Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX])
                    elif (board_positions[y][x] == 'c' and current_player == 'b'):
                        toY = y
                        hits = False
                        for toX in range(x - 1, -1, -1):
                            m = Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].isupper()):
                                        moves.append(m)
                                    break

                        hits = False
                        for toX in range(x + 1, Boardutils.Nx):
                            m = Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].isupper()):
                                        moves.append(m)
                                    break

                        toX = x
                        hits = False
                        for toY in range(y - 1, -1, -1):
                            m = Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].isupper()):
                                        moves.append(m)
                                    break

                        hits = False
                        for toY in range(y + 1, Boardutils.Ny):
                            m = Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].isupper()):
                                        moves.append(m)
                                    break
                    elif (board_positions[y][x] == 'C' and current_player == 'w'):
                        toY = y
                        hits = False
                        for toX in range(x - 1, -1, -1):
                            m = Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].islower()):
                                        moves.append(m)
                                    break

                        hits = False
                        for toX in range(x + 1, Boardutils.Nx):
                            m = Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].islower()):
                                        moves.append(m)
                                    break

                        toX = x
                        hits = False
                        for toY in range(y - 1, -1, -1):
                            m = Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].islower()):
                                        moves.append(m)
                                    break

                        hits = False
                        for toY in range(y + 1, Boardutils.Ny):
                            m = Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].islower()):
                                        moves.append(m)
                                    break
                    elif (board_positions[y][x] == 'p' and current_player == 'b'):
                        toY = y - 1
                        toX = x

                        if (Boardutils.check_bounds(toY, toX) and Boardutils.validate_move(board_positions[toY][toX], upper=False)):
                            moves.append(Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX])

                        if y < 5:
                            toY = y
                            toX = x + 1
                            if (Boardutils.check_bounds(toY, toX) and Boardutils.validate_move(board_positions[toY][toX], upper=False)):
                                moves.append(Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX])

                            toX = x - 1
                            if (Boardutils.check_bounds(toY, toX) and Boardutils.validate_move(board_positions[toY][toX], upper=False)):
                                moves.append(Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX])

                    elif (board_positions[y][x] == 'P' and current_player == 'w'):
                        toY = y + 1
                        toX = x

                        if (Boardutils.check_bounds(toY, toX) and Boardutils.validate_move(board_positions[toY][toX], upper=True)):
                            moves.append(Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX])

                        if y > 4:
                            toY = y
                            toX = x + 1
                            if (Boardutils.check_bounds(toY, toX) and Boardutils.validate_move(board_positions[toY][toX], upper=True)):
                                moves.append(Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX])

                            toX = x - 1
                            if (Boardutils.check_bounds(toY, toX) and Boardutils.validate_move(board_positions[toY][toX], upper=True)):
                                moves.append(Boardutils.board_pos_name[y][x] + Boardutils.board_pos_name[toY][toX])

        if(K_x != None and k_x != None and K_x == k_x):
            face_to_face = True
            for i in range(K_y + 1, k_y, 1):
                if(board_positions[i][K_x].isalpha()):
                    face_to_face = False

        if(face_to_face == True):
            if(current_player == 'b'):
                moves.append(Boardutils.board_pos_name[k_y][k_x] + Boardutils.board_pos_name[K_y][K_x])
            else:
                moves.append(Boardutils.board_pos_name[K_y][K_x] + Boardutils.board_pos_name[k_y][k_x])

        return moves
