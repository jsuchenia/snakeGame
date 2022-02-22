import sys
import signal
import argparse

from game import SnakeGame, DIRECTIONS
from ui import GameUi
from ai import SimpleGameAI
from gamenn import SupervisedNN


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


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser(description='Play a game with a little bit of AI')
    parser.add_argument('mode', choices=['play', 'playsimple', 'train', 'playAI'], default="play", type=str, help="Mode to play with")
    parser.add_argument('modelfile', type=str, nargs='?', help="Path to neural network model")
    args = parser.parse_args()

    flag_training = True if args.mode == "train" else False
    flag_ui = args.mode.startswith("play")
    flag_slow = True if args.mode == "play" else False

    # Game elements
    game = SnakeGame()
    simpleai = SimpleGameAI(game)
    supervised = SupervisedNN(flag_training, modelfile="models/model", layers=[20, 40])

    if flag_ui:
        ui = GameUi()
    else:
        ui = None

    while True:
        if flag_ui:
            ui.draw(game, flag_slow)

        if not flag_ui or not ui.paused:
            state = game.getstate()

            if args.mode == "play":
                step = getstep(ui, game)
            elif args.mode == "playAI":
                step = supervised.next(state)
            else:
                step = simpleai.next()

            result = game.move(step)

            if result and args.mode == "train":
                supervised.train(state, step)

                # if game.lives > 500:
                #     break

            if not result:
                print(f"Finished tour {game.lives} - max {game.maxscore}")

                if game.lives % 100 == 99:
                    supervised.save()
