# -*- coding: utf-8 -*-

test_mode = False
tie_tolerance = 40
tie_tolerance_train = 60
# tie_tolerance = 600
# tie_tolerance_train = 60
# cchess_epoches = 42

chess_en = {
    " ": "-----",
    "k": "b_gen",
    "a": "b_adv",
    "b": "b_ele",
    "n": "b_hor",
    "r": "b_cha",
    "c": "b_can",
    "p": "b_sol",
    "K": "r_gen",
    "A": "r_adv",
    "B": "r_ele",
    "N": "r_hor",
    "R": "r_cha",
    "C": "r_can",
    "P": "r_sol",
}

chess_cn = {
    " ": "一",
    "k": "将",
    "a": "士",
    "b": "象",
    "n": "馬",
    "r": "車",
    "c": "砲",
    "p": "卒",
    "K": "帅",
    "A": "仕",
    "B": "相",
    "N": "傌",
    "R": "俥",
    "C": "炮",
    "P": "兵"
}

class position:

    def __init__(self, init_x, init_y, x_or_y=False, delta=0, num=1) -> None:
        self.init_x = init_x
        self.init_y = init_y
        self.x_or_y = x_or_y
        self.delta = delta
        self.num = num
    
    def generate_position(self):
        position_ = []
        for i in range(self.num):
            position_.append([self.init_x + (i * self.delta if self.x_or_y else 0), \
                self.init_y + (0 if self.x_or_y else i * self.delta)])
        return position_


pos_of_chesses = {
    "General": [position(4, 0), position(4, 9)],
    "Soldier": [position(0, 3, True, 2, 5), position(0, 6, True, 2, 5)],
    "Cannon": [position(1, 2, True, 6, 2), position(1, 7, True, 6, 2)],
    "Advisor": [position(3, 0, True, 2, 2), position(3, 9, True, 2, 2)],
    "Elephant": [position(2, 0, True, 4, 2), position(2, 9, True, 4, 2)],
    "Horse": [position(1, 0, True, 6, 2), position(1, 9, True, 6, 2)],
    "Chariot": [position(0, 0, True, 8, 2), position(0, 9, True, 8, 2)]
}

image_path_base = "./chesses/images"
min_x = 0
max_x = 8
min_y = 0
max_y = 9

castle_x = [3, 5]
castle_y_south = [7, 9]
castle_y_north = [0, 2]

river_y = [4, 5]

class Canvas:
    overall_height = 1000 + 90
    overall_width = 900 + 80 # round(overall_height * 1.618)
    info_width = 300
    info_height = overall_height
    board_width = 800 + 80
    board_height = overall_height - 100
    board_x = round((900 + 80 - board_width)/2)
    board_y = round((overall_height - board_height)/2)
    init_size = 50
    block_len = 100 + 10
    copyright_ = "<AlphaCC Zero By ShallowMind & Tuan>---<2022.07 China·Ningbo All Rights Reserved>"
    copy_x = board_width + 200
    copy_y = overall_height - 25

canvas_scale = Canvas()
