from game import SnakeGame, GRID_SIZE, DIRECTIONS, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT
from random import randrange

class SimpleGameAI:
    def __init__(self, game: SnakeGame):
        self._game = game

    def next(self) -> int:
        pos = self._game.positions
        food = self._game.food
        direction = self._game.direction
        head = pos[0]
        idx = DIRECTIONS.index(direction)
        directions = []

        if food[0] > head[0]:
            directions.insert(0, MOVE_RIGHT)
        elif food[0] < head[0]:
            directions.insert(0, MOVE_LEFT)

        if food[1] < head[1]:
            directions.insert(0, MOVE_UP)
        elif food[1] > head[1]:
            directions.insert(0, MOVE_DOWN)

        for move in directions + DIRECTIONS:
            newpos = (head[0] + move[0], head[1] + move[1])
            nidx = DIRECTIONS.index(move)

            if newpos in pos: continue
            if newpos[0] < 0 or newpos[0] >= GRID_SIZE: continue
            if newpos[1] < 0 or newpos[1] >= GRID_SIZE: continue
            if abs(nidx - idx) == 2: continue

            return nidx - idx

        return randrange(-1, 2)
