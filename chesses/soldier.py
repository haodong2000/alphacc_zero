# -*- coding: utf-8 -*-

import chess_utils
from params import image_path_base, river_y
import chesses_base
import os
import sys

class Soldier(chesses_base.ChessPiece):

    def __init__(self, x, y, is_red, direction):
        super().__init__(x, y, is_red, direction)

    def get_chess_image(self):
        if self.is_red:
            return os.path.join(image_path_base, "red_Soldier.png")
        else:
            return os.path.join(image_path_base, "black_Soldier.png")

    def can_move(self, board, dx, dy):
        if abs(dx) + abs(dy) != 1:
            chess_utils.raise_error(__file__, sys._getframe().f_lineno, \
                message=("chess step (%d, %d) must be in (1, 0) or (0, 1) mode"%(dx, dy)))
            return False
        if (self.is_north() and dy == -1) or (self.is_south() and dy == 1):
            chess_utils.raise_error(__file__, sys._getframe().f_lineno, \
                message=("chess step (%d, %d) is invalid because of \'Cannot Go Back\'"%(dx, dy)))
            return False
        if dy == 0:
            if (self.is_north() and self.y <= river_y[0]) or (self.is_south() and self.y >= river_y[1]):
                chess_utils.raise_error(__file__, sys._getframe().f_lineno, \
                    message=("chess step (%d, %d) is invalid because of \'Cannot Move Horizontally Before Corssing River\'"%(dx, dy)))
                return False
        tx, ty = self.x + dx, self.y + dy
        if chess_utils.is_pox_in_board(tx, ty) == False:
            chess_utils.raise_error(__file__, sys._getframe().f_lineno, \
                message=("position (%d, %d) is out of the chessboard"%(tx, ty)))
            return False
        if (tx, ty) in board.chesses and board.chesses[tx, ty].is_red == self.is_red:
            chess_utils.raise_error(__file__, sys._getframe().f_lineno, \
                message=("position (%d, %d) is already occupied by yourself"%(tx, ty)))
            return False
        return True
