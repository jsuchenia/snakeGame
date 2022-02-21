import sys
import signal
import argparse

from game import SnakeGame, DIRECTIONS
from ui import GameUi
from ai import SimpleGameAI


def signal_handler() -> None:
    print('You pressed Ctrl+C!')
    sys.exit(0)


def getstep(ui: GameUi, game: SnakeGame):
    step = DIRECTIONS.index(ui.direction)
    step -= DIRECTIONS.index(game.direction)
    if step < -1:
        step += len(DIRECTIONS)
    if step > 1:
        step -= len(DIRECTIONS)

    return step


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser(description='Play a game with a little bit of AI')
    parser.add_argument('mode', choices=['play', 'train', 'playAI'], default="play", type=str)
    args = parser.parse_args()

    game = SnakeGame()
    ui = GameUi()
    simpleai = SimpleGameAI(game)
    c = 0

    while True:
        ui.draw(game)
        if not ui.paused:
            # step = getstep(ui, game)
            print(game.getstate())
            step = simpleai.next()
            if not game.move(step):
                c += 1
                print(f"Finished tour {c} - max {game.maxscore}")
