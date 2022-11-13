# -*- coding: utf-8 -*-

from chesses.advisor import Advisor
from chesses.cannon import Cannon
from chesses.chariot import Chariot
from chesses.elephant import Elephant
from chesses.general import General
from chesses.horse import Horse
from chesses.soldier import Soldier
from params import *
from chess_utils import raise_error
import sys

class Chessboard:
    chesses = dict()
    selected_chess = None

    def __init__(self, north_is_red=True) -> None:

        self.north_is_red = north_is_red

        is_red_list = [north_is_red, not north_is_red]
        direction_list = ["north", "south"]
        classes = [General, Soldier, Cannon, Advisor, Elephant, Horse, Chariot]
        names = ["General", "Soldier", "Cannon", "Advisor", "Elephant", "Horse", "Chariot"]

        for i in range(len(names)):
            for position in pos_of_chesses[names[i]]:
                poses = position.generate_position()
                for pos in poses:
                    index = 0 if pos[1] <= river_y[0] else 1
                    Chessboard.chesses[pos[0], pos[1]] = classes[i](pos[0], pos[1], \
                        is_red=is_red_list[index], direction=direction_list[index])
    
    def reload(self):
        
        north_is_red = self.north_is_red

        is_red_list = [north_is_red, not north_is_red]
        direction_list = ["north", "south"]
        classes = [General, Soldier, Cannon, Advisor, Elephant, Horse, Chariot]
        names = ["General", "Soldier", "Cannon", "Advisor", "Elephant", "Horse", "Chariot"]

        for i in range(len(names)):
            for position in pos_of_chesses[names[i]]:
                poses = position.generate_position()
                for pos in poses:
                    index = 0 if pos[1] <= river_y[0] else 1
                    Chessboard.chesses[pos[0], pos[1]] = classes[i](pos[0], pos[1], \
                        is_red=is_red_list[index], direction=direction_list[index])

    def can_move(self, x, y, dx, dy):
        if (x, y) not in self.chesses:
            raise_error(__file__, sys._getframe().f_lineno, \
                message=("(%d, %d) must be in Chessboard.chesses"%(x, y)))
            return False
        return self.chesses[x, y].can_move(self, dx, dy)

    def move(self, x, y, dx, dy):
        return self.chesses[x, y].move(self, dx, dy)
    
    def delete(self, x, y):
        del self.chesses[x, y]
    
    def select(self, x, y, player_is_red):
        # mark it selected
        if self.selected_chess == False:
            if (x, y) in self.chesses and self.chesses[x, y].is_red == player_is_red:
                self.chesses[x, y].selected = True
                self.selected_chess = self.chesses[x, y]
            return False, None
        # move it
        if (x, y) not in self.chesses:
            if self.selected_chess:
                ox, oy = self.selected_chess.x, self.selected_chess.y
                if self.can_move(ox, oy, x - ox, y - oy):
                    self.move(ox, oy, x - ox, y - oy)
                    self.chesses[x, y].selected = False
                    self.selected_chess = None
                    return True, (ox, oy, x, y)
                else:
                    raise_error(__file__, sys._getframe().f_lineno, \
                        message=("selected_chess is ready, please select leagal position"), color="yellow")
                    return False, None
            raise_error(__file__, sys._getframe().f_lineno, \
                message=("select a chess please"), color="yellow")
            return False, None
        # if is the same chess
        if self.chesses[x, y].selected == True:
            raise_error(__file__, sys._getframe().f_lineno, \
                message=("could not select the same chess: (%d, %d)"%(x, y)))
            return False, None
        # if kill happens
        if self.chesses[x, y].is_red != player_is_red:
            if self.selected_chess == None:
                raise_error(__file__, sys._getframe().f_lineno, \
                    message=("selected_chess is None, please select chess from your group first"), color="yellow")
                return False, None
            ox, oy = self.selected_chess.x, self.selected_chess.y
            if self.can_move(ox, oy, x - ox, y - oy):
                self.move(ox, oy, x - ox, y - oy)
                self.chesses[x, y].selected = False
                self.selected_chess = None
                return True, (ox, oy, x, y)
            else:
                raise_error(__file__, sys._getframe().f_lineno, \
                    message=("selected_chess is ready, please select leagal position"), color="yellow")
                return False, None
        else:
            raise_error(__file__, sys._getframe().f_lineno, \
                message=("could not kill the friend-group chess: (%d, %d)"%(x, y)), color="yellow")
        # remove selected state
        for key in self.chesses.keys():
            self.chesses[key].selected = False
        raise_error(__file__, sys._getframe().f_lineno, \
            message=("refresh, and select choosed chess"), color="yellow")
        # choose the chess
        self.chesses[x, y].selected = True
        self.selected_chess = self.chesses[x, y]
        return False, None
