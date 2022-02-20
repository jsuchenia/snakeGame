import sys
import signal

from game import SnakeGame
from ui import GameUi


def signal_handler(sig, frame) -> None:
    print('You pressed Ctrl+C!')
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    game = SnakeGame()
    ui = GameUi()
    while True:
        ui.draw(game)
        direction = ui.direction

        if not ui.paused:
            game.move(direction)
