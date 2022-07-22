# -*- coding: utf-8 -*-

import chess_utils
from params import image_path_base
import chesses_base
import os
import sys

class Horse(chesses_base.ChessPiece):

    def __init__(self, x, y, is_red, direction):
        super().__init__(x, y, is_red, direction)

    def get_chess_image(self):
        if self.is_red:
            return os.path.join(image_path_base, "red_Horse.png")
        else:
            return os.path.join(image_path_base, "black_Horse.png")

    def can_move(self, board, dx, dy):
        if min(abs(dx), abs(dy)) != 1 or max(abs(dx), abs(dy)) != 2 or (abs(dx) + abs(dy)) != 3:
            chess_utils.raise_error(__file__, sys._getframe().f_lineno, \
                message=("chess step (%d, %d) must be in (1, 2) or (2, 1) mode"%(dx, dy)))
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
        if (self.x if abs(dx) == 1 else self.x + dx/2, \
            self.y if abs(dy) == 1 else self.y + dy/2) in board.chesses:
            chess_utils.raise_error(__file__, sys._getframe().f_lineno, \
                message=("position (%d, %d) could not be reached because of \'Crappy Horse Feet\'"%(tx, ty)))
            return False
        return True
