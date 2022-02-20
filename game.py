from random import randrange

GRID_SIZE = 32
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, RIGHT, DOWN, LEFT]


class SnakeGame:
    def __init__(self, positions=None, food=None):
        self._lives = 0
        self._maxscore = 0

        if positions and food:
            self._positions = positions
            self._food = food
        else:
            self.reset()

    def reset(self):
        self._positions = [(randrange(GRID_SIZE), randrange(GRID_SIZE))]
        self._lives += 1
        self.newfood()

    def move(self, direction) -> bool:
        head = self._positions[0]
        newpos = (head[0] + direction[0], head[1] + direction[1])
        if newpos[0] < 0 or newpos[0] >= GRID_SIZE:
            self.reset()
            return False

        if newpos[1] < 0 or newpos[1] >= GRID_SIZE:
            self.reset()
            return False

        if newpos in self._positions:
            self.reset()
            return False

        self._positions.insert(0, newpos)
        if newpos == self.food:
            self.newfood()
            self._maxscore = max(self._maxscore, len(self._positions))
        else:
            self._positions.pop()
        return True

    def newfood(self):
        while True:
            nfood = (randrange(GRID_SIZE), randrange(GRID_SIZE))
            if nfood not in self.positions:
                self._food = nfood
                return
    @property
    def positions(self):
        return self._positions

    @property
    def food(self):
        return self._food

    @property
    def score(self):
        return len(self._positions)

    @property
    def maxscore(self):
        return self._maxscore

    @property
    def direction(self):
        return DIRECTIONS.index(self._direction)

    @property
    def lives(self):
        return self._lives
