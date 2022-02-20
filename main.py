import sys
import signal

from game import SnakeGame, DIRECTIONS
from ui import GameUi
from ai import SimpleGameAI


def signal_handler(sig, frame) -> None:
    print('You pressed Ctrl+C!')
    sys.exit(0)


def getstep(ui, game):
    step = DIRECTIONS.index(ui.direction)
    step -= DIRECTIONS.index(game.direction)
    if step < -1:
        step += len(DIRECTIONS)
    if step > 1:
        step -= len(DIRECTIONS)

    return step


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    game = SnakeGame()
    ui = GameUi()
    simpleai = SimpleGameAI(game)

    while True:
        ui.draw(game)
        if not ui.paused:
            # step = getstep(ui, game)
            step = simpleai.next()
            game.move(step)
