import sys
import signal
import argparse
import matplotlib.pyplot as plt

from game import SnakeGame
from ai import SimpleGameAI
from gamenn import SupervisedNN


def signal_handler(step, frame) -> None:
    print('You pressed Ctrl+C!')
    sys.exit(0)


def train(model: SupervisedNN) -> None:
    game = SnakeGame()
    simpleai = SimpleGameAI(game)
    result = True

    while result:
        state = game.getstate()
        step = simpleai.next()
        model.train(state, step)

        result = game.move(step)


def measure(model: SupervisedNN, index:int, rounds=3) -> int:
    game = SnakeGame()

    for i in range(rounds):
        result = True
        while result:
            result = game.move(model.next(game.getstate()))
        print(f" {index:6d} -> {i} - {game.maxscore}")
    return game.maxscore


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('layers', metavar='layersize', type=int, nargs='+', help='hodden size layers')
    args = parser.parse_args()

    supervised = SupervisedNN(True, prefix="experiment", layers=args.layers)
    samples = 1
    x = []
    y = []

    while True:
        train(supervised)
        samples += 1

        if samples % 100 == 0:
            x.append(samples)
            y.append(measure(supervised, samples))

        if samples > 10000:
            break

    print("Experiment finished")
    print(y)