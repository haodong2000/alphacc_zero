# -*- coding: utf-8 -*-

import chess_utils
from params import image_path_base
import chesses_base
import os
import sys

class Chariot(chesses_base.ChessPiece):

    def __init__(self, x, y, is_red, direction):
        super().__init__(x, y, is_red, direction)
    
    def get_chess_image(self):
        if self.is_red:
            return os.path.join(image_path_base, "red_Chariot.png")
        else:
            return os.path.join(image_path_base, "black_Chariot.png")

    def can_move(self, board, dx, dy):
        if dx * dy != 0:
            chess_utils.raise_error(__file__, sys._getframe().f_lineno, \
                message=("(%d, %d) must be in a straight line"%(dx, dy)))
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
        count = self.count_pieces(board, self.x, self.y, dx, dy)
        if (tx, ty) not in board.chesses:
            if count != 0:
                chess_utils.raise_error(__file__, sys._getframe().f_lineno, \
                    message=("position (%d, %d) is blocked by %d chess(es)"%(tx, ty, count)))
                return False
        else:
            if count != 0:
                chess_utils.raise_error(__file__, sys._getframe().f_lineno, \
                    message=("cannon cannot kill chess at (%d, %d), %d chess(es) in line"%(tx, ty, count)))
                return False
        return True
