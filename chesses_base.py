# -*- coding: utf-8 -*-

class ChessPiece:

    selected = False
    is_king = False
    def __init__(self, x, y, is_red, direction):
        self.x = x
        self.y = y
        self.is_red = is_red
        self.direction = direction

    def is_north(self):
        return self.direction == 'north'

    def is_south(self):
        return self.direction == 'south'

    def get_move_locs(self, board):
        moves = []
        for x in range(9):
            for y in range(10):
                if (x,y) in board.chesses and board.chesses[x,y].is_red == self.is_red:
                    continue
                if self.can_move(board, x-self.x, y-self.y):
                    moves.append((x,y))
        return moves

    def move(self, board, dx, dy):
        nx, ny = self.x + dx, self.y + dy
        if (nx, ny) in board.chesses:
            board.delete(nx, ny)
        board.delete(self.x, self.y)
        # print('Move a chessman from (%d,%d) to (%d,%d)'%(self.x, self.y, self.x+dx, self.y+dy))
        self.x += dx
        self.y += dy
        board.chesses[self.x, self.y] = self
        return True

    def count_pieces(self, board, x, y, dx, dy):
        sx = dx/abs(dx) if dx!=0 else 0
        sy = dy/abs(dy) if dy!=0 else 0
        nx, ny = x + dx, y + dy
        x, y = x + sx, y + sy
        cnt = 0
        while x != nx or y != ny:
            if (x, y) in board.chesses:
                cnt += 1
            x += sx
            y += sy
        return cnt
