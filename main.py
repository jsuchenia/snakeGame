import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from random import randrange, choice
import pygame
import sys
import torch

GRID_SIZE = 32
CELL_SIZE = 15
SCREEN_SIZE = GRID_SIZE * CELL_SIZE

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

DIRECTIONS = [UP, RIGHT, DOWN, LEFT]


class SnakeGame:
    def __init__(self, positions=None, direction=None, food=None):
        self._lives = 0

        if positions and direction and food:
            self._positions = positions
            self._direction = direction
            self._food = food
        else:
            self.reset()

    def reset(self):
        self._positions = [(randrange(GRID_SIZE), randrange(GRID_SIZE))]
        self._direction = choice(DIRECTIONS)
        self._lives += 1
        self.newfood()

    def move(self, direction=None):
        if direction:
            d = direction
        else:
            d = self._direction

        head = self._positions[0]
        newpos = (head[0] + d[0], head[1] + d[1])
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
    def direction(self):
        return DIRECTIONS.index(self._direction)

    @direction.setter
    def direction(self, direction):
        cur = DIRECTIONS.index(self._direction)
        new = DIRECTIONS.index(direction)

        if abs(cur-new) != 2:
            self._direction = direction

    @property
    def lives(self):
        return self._lives


if __name__ == '__main__':
    snake = SnakeGame()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    pygame.init()
    pygame.display.set_caption("Snake Game - Stefan")
    screen = pygame.display.set_mode([SCREEN_SIZE, SCREEN_SIZE+100])
    gameSurface = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE+100))

    font = pygame.font.SysFont('Arial', 25)

    clock = pygame.time.Clock()

    maxscore = 0

    while True:
        clock.tick(5)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                snake.direction = UP
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                snake.direction = DOWN
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                snake.direction = LEFT
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                snake.direction = RIGHT

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if ((x + y) % 2) == 0:
                    pygame.draw.rect(gameSurface, (0, 190, 0),
                                     pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                else:
                    pygame.draw.rect(gameSurface, (0, 170, 0),
                                     pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for s in snake.positions:
            pygame.draw.rect(gameSurface, (0, 0, 200), pygame.Rect((s[0] * CELL_SIZE, s[1] * CELL_SIZE), (CELL_SIZE, CELL_SIZE)))

        food = snake.food
        maxscore = max(maxscore, snake.score)
        pygame.draw.rect(gameSurface, (250, 0, 0), pygame.Rect((food[0] * CELL_SIZE, food[1] * CELL_SIZE), (CELL_SIZE, CELL_SIZE)))
        pygame.draw.rect(gameSurface, (0, 0, 0), pygame.Rect((0, SCREEN_SIZE), (SCREEN_SIZE, 100)))
        gameSurface.blit(font.render(f"Points: {snake.score}", False, (255, 255, 255), (0, 0, 0)), (5, SCREEN_SIZE))
        gameSurface.blit(font.render(f"Lives: {snake.lives}", False, (255, 255, 255), (0, 0, 0)), (SCREEN_SIZE - 110, SCREEN_SIZE))
        gameSurface.blit(font.render(f"Max: {maxscore}", False, (255, 255, 255), (0, 0, 0)), (5, SCREEN_SIZE + 30))
        screen.blit(gameSurface, pygame.Rect(0, 0, SCREEN_SIZE, SCREEN_SIZE))
        pygame.display.flip()

        snake.move()
    pygame.quit()