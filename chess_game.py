# -*- coding: utf-8 -*-

from chess_board import Chessboard
from main import *
import tkinter
from visualization import Visualization
from params import *

class Chessgame:

    board = None
    cur_round = 1
    game_mode = 1
    ''' 
    0 -> HUMAN V.S. HUMAN 
    1 -> HUMAN V.S. AI 
    2 -> AI V.S. AI
    '''
    time_red = []
    time_black = []
    model_path = "./gpu_models/checkpoints/ckpt-"

    def __init__(self, in_ai_count, in_ai_function, in_play_playout, in_delay, in_end_delay, batch_size, search_threads,
                 processor, num_gpus, res_block_nums, human_color="b", if_eval=False):
        self.human_color = human_color
        self.current_player = "w"
        self.players = {}
        self.players[self.human_color] = "human"
        ai_color = "w" if self.human_color == "b" else "b"
        self.players[ai_color] = "AI"

        Chessgame.board = Chessboard(self.human_color == 'b')
        self.view = Visualization(self, board=Chessgame.board)
        self.view.show_msg("Initialization ...")
        self.view.draw_board(self.board)
        Chessgame.game_mode = in_ai_count
        self.ai_function = in_ai_function
        self.play_playout = in_play_playout
        self.delay = in_delay
        self.end_delay = in_end_delay

        self.win_rate = {}
        self.win_rate['w'] = 0.0
        self.win_rate['b'] = 0.0

        self.view.root.update()
        self.cchess_engine = cchess_main(playout=self.play_playout, in_batch_size=batch_size, exploration=False, in_search_threads=search_threads,
                                         processor=processor, num_gpus=num_gpus, res_block_nums=res_block_nums, human_color=human_color)

        self.if_eval = if_eval
        self.eval_num = 0

    def player_is_red(self):
        return self.current_player == "w"

    def start(self):
        self.view.show_msg("Red")
        if self.game_mode == 1:
            print ("-" * 40 + '<Round %d>' % self.cur_round + "-" * 40)
            if self.players["w"] == "AI":
                self.win_rate['w'] = self.perform_AI()
                self.view.draw_board(self.board)
                self.change_player()
        elif self.game_mode == 2:
            print ("-" * 40 + '<Round %d>' % self.cur_round + "-" * 40)
            self.win_rate['w'] = self.perform_AI()
            self.view.draw_board(self.board)

        self.view.start()

    def eval_start(self, eval_num):
        self.cchess_engine.game_borad.reload()
        self.cchess_engine.mcts.reload()
        self.board.reload()
        self.view.board = self.board
        self.view.show_msg("Initialization ...")
        self.view.draw_board(self.board)
        self.current_player = 'w'
        self.cur_round = 1

        # time.sleep(1)
        self.eval_num = eval_num

        self.view.show_msg("Red")

        print ("-" * 40 + '<Round %d>' % self.cur_round + "-" * 40)

        if self.if_eval:
            self.cchess_engine.policy_value_netowrk.checkpoint.restore(self.model_path + str(eval_num))

        self.win_rate['w'] = self.perform_AI()
        self.view.draw_board(self.board)

        self.view.start()

        ret, winner = self.cchess_engine.check_end()

        if ret:
            return winner
        else:
            return 'error'

    def disp_mcts_msg(self):
        self.view.show_msg("MCTS Searching...")

    def callback(self, event):
        # triggered once the mouse button is pushed
        if self.game_mode == 1 and self.players[self.current_player] == "AI":
            return
        if self.game_mode == 2:
            return
        rx, ry = self.real_coord(event.x), self.real_coord(event.y)
        change, coord = self.board.select(rx, ry, self.player_is_red())
        self.view.draw_board(self.board)
        if self.check_end():
            self.view.root.update()
            self.quit()
            return
        if change:
            self.view.draw_board(self.board)
            self.win_rate[self.current_player] = self.cchess_engine.human_move(coord, self.ai_function)
            if self.check_end():
                self.view.root.update()
                self.quit()
                return
            performed = self.change_player()
            if performed:
                self.view.draw_board(self.board)
                if self.check_end():
                    self.view.root.update()
                    self.quit()
                    return
                self.change_player()

    def board_coord(self, x):
        return canvas_scale.init_size + canvas_scale.block_len * x

    def real_coord(self, x):
        if x <= canvas_scale.init_size:
            return 0
        else:
            return (x - canvas_scale.init_size - canvas_scale.init_size)//canvas_scale.block_len + 1

    def quit(self):
        time.sleep(self.end_delay)
        self.view.quit()

    def check_end(self):
        ret, winner = self.cchess_engine.check_end()
        if ret == True:
            if winner == "b":
                self.view.show_msg(' Black Wins at Round %d ' % self.cur_round)
                self.view.root.update()
            elif winner == "w":
                self.view.show_msg(' Red Wins at Round %d ' % self.cur_round)
                self.view.root.update()
            elif winner == "t":
                self.view.show_msg(' Draw at Round %d ' % self.cur_round)
                self.view.root.update()
        return ret

    def change_player(self):
        self.current_player = "w" if self.current_player == "b" else "b"

        if self.if_eval:
            if self.current_player == "w":
                self.cchess_engine.policy_value_netowrk.checkpoint.restore(self.model_path + str(self.eval_num))
            else:
                self.cchess_engine.policy_value_netowrk.checkpoint.restore(self.model_path + str(self.eval_num + 1))

        if self.current_player == "w":
            self.cur_round += 1
            print ("-" * 40 + '<Round %d>' % self.cur_round + "-" * 40)
        red_msg = " ({:.4f})".format(self.win_rate['w'])
        green_msg = " ({:.4f})".format(self.win_rate['b'])
        sorted_move_probs = self.cchess_engine.get_hint(self.ai_function, True, self.disp_mcts_msg)
        self.view.print_all_hint(sorted_move_probs)

        self.view.show_msg("Red" + red_msg + " Black" + green_msg if self.current_player == "w" else "Black" + green_msg + " Red" + red_msg)
        self.view.root.update()

        if self.game_mode == 1:
            if self.players[self.current_player] == "AI":
                self.win_rate[self.current_player] = self.perform_AI()
                return True
            return False
        elif self.game_mode == 2:
            self.win_rate[self.current_player] = self.perform_AI()
            return True
        return False

    def perform_AI(self):
        print ("=" * 40 + '<AI Calculating...>' + "=" * 40)
        start_time = time.time()
        move, win_rate = self.cchess_engine.select_move(self.ai_function)
        time_used = time.time() - start_time
        print ("=" * 40  + '<AI Took %fs>' % time_used + "=" * 40)
        if self.current_player == "w":
            self.time_red.append(time_used)
        else:
            self.time_black.append(time_used)
        if move is not None:
            self.board.move(move[0], move[1], move[2], move[3])
        return win_rate

    # AI VS AI mode
    def game_mode_2(self):
        self.change_player()
        self.view.draw_board(self.board)
        self.view.root.update()
        if self.check_end():
            return True
        return False
