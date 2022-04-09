import sys
import signal
import argparse

from game.game import SnakeGame, DIRECTIONS
from game.ui import GameUi
from game.ai import SimpleGameAI
from game.gamenn import SupervisedNN


def signal_handler(step, frame) -> None:
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


def getlayersfrompath(modelfile):
    idx = modelfile.find("-")
    return [int(x) for x in modelfile[idx+1:].split("-")]


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser(description='Play a game with a little bit of AI')
    parser.add_argument('mode', choices=['play', 'playsimple', 'playAI'], default="play", type=str, help="Mode to play with")
    parser.add_argument('modelfile', type=str, nargs='?', help="Path to neural network model")
    args = parser.parse_args()

    flag_slow = True if args.mode == "play" else False

    if args.modelfile:
        modelfile = args.modelfile
        layers = getlayersfrompath(modelfile)
    else:
        modelfile="models/model"
        layers=[20, 40]

    # Game elements
    game = SnakeGame()
    simpleai = SimpleGameAI(game)
    supervised = SupervisedNN(False, modelfile=modelfile, layers=layers)

    ui = GameUi()

    while True:
        ui.draw(game, flag_slow)

        if not ui.paused:
            state = game.getstate()

            if args.mode == "play":
                step = getstep(ui, game)
            elif args.mode == "playAI":
                step = supervised.next(state)
            else:
                step = simpleai.next()

            result = game.move(step)

            # if game.lives > 500:
            #     break

            if not result:
                print(f"Finished tour {game.lives} - max {game.maxscore}")

                if game.lives % 100 == 99:
                    supervised.save()
