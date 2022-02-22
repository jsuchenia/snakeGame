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
    last_step = 1

    while result:
        state = game.getstate()
        step = simpleai.next()

        if step !=0 or last_step != 0:
            model.train(state, step)

        result = game.move(step)
        last_step = step
    # print(game.maxscore)


def measure(model: SupervisedNN, index:int, rounds=3) -> int:
    game = SnakeGame()

    results = []
    for i in range(rounds):
        result = True
        while result:
            result = game.move(model.next(game.getstate()))
        results.append(game.maxscore)
    print(f" {index:6d} -> {results}")
    return game.maxscore


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('layers', metavar='layersize', type=int, nargs='+', help='hodden size layers')
    args = parser.parse_args()

    suffix = '-'.join([str(layer) for layer in args.layers])
    modelfile = f"models/experiment-{suffix}"

    supervised = SupervisedNN(True, modelfile=modelfile, layers=args.layers)
    samples = 1
    x = []
    y = []

    supervised.trainmode()
    while True:
        train(supervised)
        samples += 1

        if samples % 100 == 0:
            supervised.save()
            supervised.usemode()
            x.append(samples)
            y.append(measure(supervised, samples))
            supervised.trainmode()

        if samples > 10000:
            break

    print("Experiment finished")
    print(y)