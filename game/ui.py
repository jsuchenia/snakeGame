import os
import sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from game.game import GRID_SIZE, DIRECTIONS, MOVE_UP, MOVE_LEFT, MOVE_DOWN, MOVE_RIGHT


class GameUi:
    CELL = 15
    SCREEN = GRID_SIZE * CELL

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Snake Game - Stefan")
        self.screen = pygame.display.set_mode([self.SCREEN, self.SCREEN + 100])
        self.surface = pygame.Surface((self.SCREEN, self.SCREEN + 100))
        self.font = pygame.font.SysFont('Arial', 25)
        self.clock = pygame.time.Clock()
        self.move_direction = MOVE_UP
        self._paused = False

    def draw_background(self) -> None:
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if ((x + y) % 2) == 0:
                    pygame.draw.rect(self.surface, (0, 190, 0),
                                     pygame.Rect(x * self.CELL, y * self.CELL, self.CELL, self.CELL))
                else:
                    pygame.draw.rect(self.surface, (0, 170, 0),
                                     pygame.Rect(x * self.CELL, y * self.CELL, self.CELL, self.CELL))

    def draw_gameelements(self, game) -> None:
        food = game.food
        for s in game.positions:
            pygame.draw.rect(self.surface, (0, 0, 200),
                             pygame.Rect((s[0] * self.CELL, s[1] * self.CELL), (self.CELL, self.CELL)))
        pygame.draw.rect(self.surface, (250, 0, 0),
                         pygame.Rect((food[0] * self.CELL, food[1] * self.CELL), (self.CELL, self.CELL)))

    def draw_summary(self, game) -> None:
        pygame.draw.rect(self.surface, (0, 0, 0), pygame.Rect((0, self.SCREEN), (self.SCREEN, 100)))
        self.surface.blit(self.font.render(f"Points: {game.score}", False, (255, 255, 255), (0, 0, 0)), (5, self.SCREEN))
        self.surface.blit(self.font.render(f"Lives: {game.lives}", False, (255, 255, 255), (0, 0, 0)),
                          (self.SCREEN/2, self.SCREEN))
        self.surface.blit(self.font.render(f"Max: {game.maxscore}", False, (255, 255, 255), (0, 0, 0)), (5, self.SCREEN + 30))
        self.surface.blit(self.font.render(f"FPS: {self.clock.get_fps():.2f}", False, (255, 255, 255), (0, 0, 0)), (self.SCREEN/2, self.SCREEN + 30))

        if self.paused:
            self.surface.blit(self.font.render("Paused", False, (255, 255, 255), (0, 0, 0)), (250, self.SCREEN + 60))

    def set_direction(self, new) -> None:
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
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                self.close()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self._paused = not self._paused
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.set_direction(MOVE_UP)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.set_direction(MOVE_DOWN)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.set_direction(MOVE_LEFT)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.set_direction(MOVE_RIGHT)

    @property
    def direction(self):
        return self.move_direction

    @property
    def paused(self):
        return self._paused

    def draw(self, game, slow=False) -> None:
        self.clock.tick(10 if slow else 0)

        self.check_events()
        self.draw_background()
        self.draw_gameelements(game)
        self.draw_summary(game)

        self.screen.blit(self.surface, pygame.Rect(0, 0, self.SCREEN, self.SCREEN))
        pygame.display.flip()
