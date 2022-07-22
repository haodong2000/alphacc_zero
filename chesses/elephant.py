# -*- coding: utf-8 -*-

import chess_utils
from params import image_path_base, river_y
import chesses_base
import os
import sys

class Elephant(chesses_base.ChessPiece):

    def __init__(self, x, y, is_red, direction):
        super().__init__(x, y, is_red, direction)
    
    def get_chess_image(self):
        if self.is_red:
            return os.path.join(image_path_base, "red_Elephant.png")
        else:
            return os.path.join(image_path_base, "black_Elephant.png")
    
    def is_corss_river(self, tx, ty):
        if (self.is_north() and ty <= river_y[0]) or \
            (self.is_south() and ty >= river_y[1]):
            return True
        else:
            return False

    def can_move(self, board, dx, dy):
        if abs(dx) != 2 or abs(dy) != 2:
            chess_utils.raise_error(__file__, sys._getframe().f_lineno, \
                message=("chess step (%d, %d) must be in (2, 2) mode"%(dx, dy)))
            return False
        tx, ty = self.x + dx, self.y + dy
        mid_x, mid_y = dx/abs(dx), dy/abs(dy)
        if (self.x + mid_x, self.y + mid_y) in board.chesses:
            chess_utils.raise_error(__file__, sys._getframe().f_lineno, \
                message=("position (%d, %d) could not be reached because of \'Crappy Elephant Feet\'"%(tx, ty)))
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
        if self.is_corss_river(tx, ty) == False:
            chess_utils.raise_error(__file__, sys._getframe().f_lineno, \
                message=("position (%d, %d) will corss the river"%(tx, ty)))
            return False
        return True
