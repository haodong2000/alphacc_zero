# -*- coding: utf-8 -*-

import sys
import params

def is_pox_in_board(x, y):
    if x < params.min_x or \
        x > params.max_x or \
            y < params.min_y or \
                y > params.max_y:
                return False
    else:
        return True

def raise_error(file, line, message ,color="red"):
    if color == "red":
        pass
        # print('\033[31m<USER ERROR> <File: %s> <Line: %s>\n\t%s<Msg: %s>\033[0m'%(file, line, " "*5, message))
    else:
        print('\033[1;33;1m<USER ERROR> <File: %s> <Line: %s>\n\t%s<Msg: %s>\033[0m'%(file, line, " "*5, message))

if __name__ == "__main__":
    raise_error(__file__, sys._getframe().f_lineno, \
            message=("(%d, %d) is out of the chessboard"%(9, 9)))
