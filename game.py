import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from random import randrange, choice
import pygame
import sys
import signal
import torch
import numpy as np

GRID_SIZE = 32
CELL_SIZE = 15
SCREEN_SIZE = GRID_SIZE * CELL_SIZE

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

    def move(self, direction):
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


class GameUi:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Snake Game - Stefan")
        self.screen = pygame.display.set_mode([SCREEN_SIZE, SCREEN_SIZE + 100])
        self.surface = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE + 100))
        self.font = pygame.font.SysFont('Arial', 25)
        self.clock = pygame.time.Clock()
        self.move_direction = UP

    def draw_background(self):
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if ((x + y) % 2) == 0:
                    pygame.draw.rect(self.surface, (0, 190, 0),
                                     pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                else:
                    pygame.draw.rect(self.surface, (0, 170, 0),
                                     pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def draw_gameelements(self, game):
        for s in game.positions:
            pygame.draw.rect(self.surface, (0, 0, 200),
                             pygame.Rect((s[0] * CELL_SIZE, s[1] * CELL_SIZE), (CELL_SIZE, CELL_SIZE)))
        food = game.food
        pygame.draw.rect(self.surface, (250, 0, 0),
                         pygame.Rect((food[0] * CELL_SIZE, food[1] * CELL_SIZE), (CELL_SIZE, CELL_SIZE)))

    def draw_summary(self, game):
        pygame.draw.rect(self.surface, (0, 0, 0), pygame.Rect((0, SCREEN_SIZE), (SCREEN_SIZE, 100)))
        self.surface.blit(self.font.render(f"Points: {game.score}", False, (255, 255, 255), (0, 0, 0)), (5, SCREEN_SIZE))
        self.surface.blit(self.font.render(f"Lives: {game.lives}", False, (255, 255, 255), (0, 0, 0)),
                         (SCREEN_SIZE - 110, SCREEN_SIZE))
        self.surface.blit(self.font.render(f"Max: {game.maxscore}", False, (255, 255, 255), (0, 0, 0)), (5, SCREEN_SIZE + 30))

    def set_direction(self, new):
        c = DIRECTIONS.index(self.move_direction)
        n = DIRECTIONS.index(new)
        if abs(c-n) != 2:
            self.move_direction = new

    def close(self) -> None:
        pygame.quit()
        sys.exit(0)

    def check_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
               self.close()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.set_direction(UP)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.set_direction(DOWN)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.set_direction(LEFT)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.set_direction(RIGHT)

    @property
    def direction(self):
        return self.move_direction

    def draw(self, game):
        self.clock.tick(8)

        self.check_events()

        self.draw_background()
        self.draw_gameelements(game)
        self.draw_summary(game)

        self.screen.blit(self.surface, pygame.Rect(0, 0, SCREEN_SIZE, SCREEN_SIZE))
        pygame.display.flip()


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    game = SnakeGame()
    ui = GameUi()
    while True:
        ui.draw(game)
        game.move(ui.direction)
