from random import randrange
import typing

GRID_SIZE = 48

MOVE_UP = (0, -1)
MOVE_UPRIGHT = (1, -1)
MOVE_RIGHT = (1, 0)
MOVE_DOWNRIGHT = (1, 1)
MOVE_DOWN = (0, 1)
MOVE_DOWNLEFT = (-1, 1)
MOVE_LEFT = (-1, 0)
MOVE_UPLEFT = (-1, -1)

STEP_LEFT = -1
STEP_STRIGHT = 0
STEP_RIGHT = 1
DIRECTIONS = [MOVE_UP, MOVE_RIGHT, MOVE_DOWN, MOVE_LEFT]


class SnakeGameState(typing.NamedTuple):
    headx: int
    heady: int
    foodx: int
    foody: int
    size: int
    direction: int
    onleft: bool
    onstraight: bool
    onright: bool


class SnakeGame:
    def __init__(self):
        self._lives = 0
        self._maxscore = 0
        self._positions = []
        self._direction = MOVE_UP
        self._food = (0, 0)
        self._steps = 0
        self.reset()

    def reset(self):
        self._positions = [(randrange(GRID_SIZE), randrange(GRID_SIZE))]
        self._lives += 1
        self._steps = 0
        self._direction = MOVE_UP
        self.newfood()

    def setdirection(self, step):
        idx = DIRECTIONS.index(self._direction) + step
        idx += len(DIRECTIONS)
        idx %= len(DIRECTIONS)
        self._direction = DIRECTIONS[idx]

    def move(self, step) -> bool:
        self.setdirection(step)
        head = self._positions[0]
        newpos = (head[0] + self._direction[0], head[1] + self._direction[1])
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
            self._steps = 0
            self.newfood()
            self._maxscore = max(self._maxscore, len(self._positions))
        else:
            self._positions.pop()
            self._steps += 1

        if self._steps > 10*GRID_SIZE:
            self.reset()
            return False
        return True

    def newfood(self):
        while True:
            nfood = (randrange(GRID_SIZE), randrange(GRID_SIZE))
            if nfood not in self.positions:
                self._food = nfood
                return

    def _iscellvalid(self, x:int , y:int) -> bool:
        if x < 0 or x >= GRID_SIZE:
            return False

        if y < 0 or y >= GRID_SIZE:
            return False

        if (x, y) in self._positions:
            return False
        return True

    def _checkdirection(self, offset:int) -> bool:
        head = self._positions[0]
        direction = DIRECTIONS.index(self._direction)
        newoffset = direction + offset
        if newoffset < 0: newoffset += len(DIRECTIONS)
        if newoffset >= len(DIRECTIONS): newoffset -= len(DIRECTIONS)

        diff = DIRECTIONS[newoffset]

        return self._iscellvalid(head[0] + diff[0], head[1] + diff[1])

    def getstate(self):
        head = self._positions[0]

        return SnakeGameState(
            headx=head[0],
            heady=head[1],
            foodx=self._food[0],
            foody=self._food[1],
            size=len(self._positions),
            direction=DIRECTIONS.index(self._direction),
            onleft=self._checkdirection(-1),
            onstraight=self._checkdirection(0),
            onright=self._checkdirection(1)
        )

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
        return self._direction

    @property
    def lives(self):
        return self._lives
