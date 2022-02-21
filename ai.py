from game import SnakeGame, GRID_SIZE, DIRECTIONS, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT
from random import randrange

class SimpleGameAI:
    def __init__(self, game: SnakeGame):
        self._game = game

    def next(self) -> int:
        state = self._game.getstate()
        directions = []

        if state.foodx > state.headx:
            directions.insert(0, MOVE_RIGHT)
        elif state.foodx < state.headx:
            directions.insert(0, MOVE_LEFT)

        if state.foody < state.heady:
            directions.insert(0, MOVE_UP)
        elif state.foody > state.heady:
            directions.insert(0, MOVE_DOWN)

        for move in directions + DIRECTIONS:
            newpos = (state.headx + move[0], state.heady + move[1])
            nidx = DIRECTIONS.index(move)

            if newpos in self._game.positions: continue
            if newpos[0] < 0 or newpos[0] >= GRID_SIZE: continue
            if newpos[1] < 0 or newpos[1] >= GRID_SIZE: continue

            diff = nidx - state.direction
            if abs(diff) == 2: continue

            if diff < -1:
                diff += len(DIRECTIONS)

            if diff > 1:
                diff -= len(DIRECTIONS)

            return diff

        return randrange(-1, 2)
