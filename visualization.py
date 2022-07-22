# -*- coding: utf-8 -*-

import tkinter
from tkinter import font
import time
from chess_board import Chessboard
from PIL import Image, ImageTk
from params import *
from datetime import datetime


class Visualization:
    root = tkinter.Tk()
    root.title("AlphaCC Zero UI -> Initialization")
    root.resizable(height=False, width=False)
    # root.geometry("%dx%d"%(canvas_scale.overall_width + canvas_scale.info_width, canvas_scale.overall_height))
    canvas = tkinter.Canvas(root, width=canvas_scale.overall_width, height=canvas_scale.overall_height, \
                            background="white")
    canvas.pack(expand=tkinter.YES, fill=tkinter.BOTH)
    board_img = Image.open("./chesses/images/board.png")
    board_img = board_img.resize((canvas_scale.board_width, canvas_scale.board_height), \
                                 Image.LANCZOS)
    board_img = ImageTk.PhotoImage(board_img)
    canvas.create_image(canvas_scale.board_x, canvas_scale.board_y, \
                        image=board_img, anchor=tkinter.NW)
    piece_images = dict()
    move_images = []
    # copy_font = font.Font(family='Arial', size=10, weight='bold')
    # copy = tkinter.Label(root, text=canvas_scale.copyright_, fg="black", bg="white", \
    #                      font=copy_font)
    # copy.place(x=canvas_scale.copy_x, y=canvas_scale.copy_y)
    # label_font = font.Font(family='Times 20 italic bold', size=20, weight='bold')
    # lb_label = tkinter.Label(root, text="Sorted Move Probilities", fg="black", bg="white", \
    #                          font=label_font)
    # lb_label.place(x=canvas_scale.copy_x, y=canvas_scale.overall_height - canvas_scale.copy_y)

    def print_all_hint(self, sorted_move_probs):
        # self.lb.delete(0, "end")
        print("-" * 90)
        for item in sorted_move_probs:
            print(str(item) + datetime.now().strftime(' ----- %Y-%m-%d %H:%M:%S'))
            # self.lb.insert("end", str(item) + datetime.now().strftime(' ----- %Y-%m-%d %H:%M:%S'))
        print("-" * 90)
        # self.lb.pack()

    def print_list(self, event):
        w = event.widget
        index = round(w.curselection()[0]) # 2022-07-22
        value = w.get(index)
        self.disp_hint_on_board(value[0], value[1])

    def __init__(self, control, board) -> None:
        self.control = control
        if self.control.game_mode != 2:
            self.canvas.bind("<Button-1>", self.control.callback) # signal slot

        self.lb = tkinter.Listbox(Visualization.root, selectmode="browse", bg="pink", bd=0, fg="black", \
                                  cursor="target", justify="center", height=0)
        # self.scr1 = tkinter.Scrollbar(Visualization.root, bg="cyan", command=self.lb.yview, width=0)
        # self.lb.configure(yscrollcommand=self.scr1.set)
        # self.scr1.pack(padx=0, pady=0, side='right',fill="y")
        # self.lb.pack(fill="x")

        self.lb.bind('<<ListboxSelect>>', self.print_list) # signal slot
        self.board = board
        self.last_text_x = 0
        self.last_text_y = 0
        self.print_text_flag = False

    def start(self):
        if self.control.game_mode == 2:
            self.root.update()
            time.sleep(self.control.delay)
            while True:
                game_end = self.control.game_mode_2()
                self.root.update()
                time.sleep(self.control.delay)
                if game_end:
                    # game_end is True once the check_end() returns True
                    time.sleep(self.control.end_delay)
                    self.quit()
                    return
        else:
            tkinter.mainloop()

    def board_coord(self, x):
        return canvas_scale.init_size + canvas_scale.block_len * x

    def draw_board(self, board):
        self.piece_images.clear()
        self.move_images = []
        pieces = board.chesses
        for (x, y) in pieces.keys():
            self.piece_images[x, y] = tkinter.PhotoImage(file=pieces[x, y].get_chess_image())
            self.canvas.create_image(self.board_coord(x), self.board_coord(y), image=self.piece_images[x, y])
        if self.board.selected_chess:
            self.move_images.append(tkinter.PhotoImage(file="./chesses/images/target.png"))
            self.canvas.create_image(self.board_coord(self.board.selected_chess.x), self.board_coord(self.board.selected_chess.y), image=self.move_images[-1])
            for (x, y) in self.board.selected_chess.get_move_locs(self.board):
                self.move_images.append(tkinter.PhotoImage(file="./chesses/images/target.png"))
                self.canvas.create_image(self.board_coord(x), self.board_coord(y), image=self.move_images[-1])
        self.root.update() # 2022-07-22

    def show_msg(self, msg):
        self.root.title("AlphaCC Zero UI -> " + msg)

    def disp_hint_on_board(self, action, percentage):
        board = self.board
        for key in board.chesses.keys():
            board.chesses[key].selected = False
        board.selected_piece = None

        self.canvas.create_image(canvas_scale.board_x, canvas_scale.board_y, \
                                 image=self.board_img, anchor=tkinter.NW)
        self.draw_board(board)

        x_trans = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}

        src = action[0:2]
        dst = action[2:4]

        src_x = round(x_trans[src[0]]) # 2022-07-22
        src_y = round(src[1])

        dst_x = round(x_trans[dst[0]]) # 2022-07-22
        dst_y = round(dst[1])

        pieces = board.chesses
        if (src_x, src_y) in pieces.keys():
            self.piece_images[src_x, src_y] = tkinter.PhotoImage(file=pieces[src_x, src_y].get_chess_image())
            self.canvas.create_image(self.board_coord(src_x), self.board_coord(src_y), image=self.piece_images[src_x, src_y])

        if (dst_x, dst_y) in pieces.keys():
            self.piece_images[dst_x, dst_y] = tkinter.PhotoImage(file=pieces[dst_x, dst_y].get_chess_image())
            self.canvas.create_image(self.board_coord(dst_x), self.board_coord(dst_y), image=self.piece_images[dst_x, dst_y])
            self.canvas.create_text(self.board_coord(dst_x), self.board_coord(dst_y), text=percentage)
            self.last_text_x = dst_x
            self.last_text_y = dst_y
        else:
            self.move_images.append(tkinter.PhotoImage(file="./chesses/images/target.png"))
            self.canvas.create_image(self.board_coord(dst_x), self.board_coord(dst_y), image=self.move_images[-1])
            self.canvas.create_text(self.board_coord(dst_x), self.board_coord(dst_y), text=percentage)
            self.last_text_x = dst_x
            self.last_text_y = dst_y
            self.print_text_flag = True

    def quit(self):
        self.root.quit()


if __name__ == '__main__':
    if test_mode:
        view = Visualization(Chessboard(north_is_red=False))
        view.draw_board()
        view.start()
