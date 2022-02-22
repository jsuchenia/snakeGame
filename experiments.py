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


def train(model: SupervisedNN, skipduplicates:True) -> None:
    game = SnakeGame()
    simpleai = SimpleGameAI(game)
    result = True
    last_step = 1

    while result:
        state = game.getstate()
        step = simpleai.next()

        if not skipduplicates or step !=0 or last_step != 0:
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
    parser.add_argument('--batch', type=int, default=10, help="Traiing batch size")
    parser.add_argument('--samples', type=int, default=10000, help="Training samples")
    parser.add_argument('--skip', action=argparse.BooleanOptionalAction, help="Skip duplicates")
    parser.add_argument('layers', metavar='layersize', type=int, nargs='+', help='hodden size layers')
    args = parser.parse_args()

    suffix = '-'.join([str(layer) for layer in args.layers])
    skip = "skip" if args.skip else "noskip"
    modelfile = f"models/experiment_batch{args.batch}_samples{args.samples}_{skip}-{suffix}"

    supervised = SupervisedNN(True, modelfile=modelfile, batchsize=args.batch, layers=args.layers)
    samples = 1
    x = []
    y = []

    supervised.trainmode()
    while True:
        train(supervised, args.skip)
        samples += 1

        if samples % 100 == 0:
            supervised.save()
            supervised.usemode()
            x.append(samples)
            y.append(measure(supervised, samples))
            supervised.trainmode()

        if samples > args.samples:
            break

    title = f"Experiment batch={args.batch} samples={args.samples} "
    title += " x ".join([str(x) for x in args.layers])

    plt.xlabel("Number of samples")
    plt.ylabel("Max achieved points")
    plt.plot(x, y)
    plt.savefig("out/experiment-" + suffix)