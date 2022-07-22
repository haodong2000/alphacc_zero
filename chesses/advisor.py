# -*- coding: utf-8 -*-

from ctypes import util
from email.mime import image
import chesses_base
import os
from params import image_path_base
from params import castle_x, castle_y_north, castle_y_south
import os, sys
import chess_utils

class Advisor(chesses_base.ChessPiece):

    def __init__(self, x, y, is_red, direction):
        super().__init__(x, y, is_red, direction)
    
    def get_chess_image(self):
        if self.is_red:
            return os.path.join(image_path_base, "red_Advisor.png")
        else:
            return os.path.join(image_path_base, "black_Advisor.png")
        
    def is_in_castle(self, tx, ty):
        if (self.is_north() and castle_x[0] <= tx <= castle_x[1] and castle_y_north[0] <= ty <= castle_y_north[1]) or \
                (self.is_south() and castle_x[0] <= tx <= castle_x[1] and castle_y_south[0] <= ty <= castle_y_south[1]):
            return True
        else:
            return False
    
    def can_move(self, board, dx, dy):
        if abs(dx) != 1 or abs(dy) != 1:
            chess_utils.raise_error(__file__, sys._getframe().f_lineno, \
                message=("chess step (%d, %d) is not diagonal"%(dx, dy)))
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
        if self.is_in_castle(tx, ty) == False:
            chess_utils.raise_error(__file__, sys._getframe().f_lineno, \
                message=("position (%d, %d) is out of castle"%(tx, ty)))
            return False
        return True
