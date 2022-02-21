from game import SnakeGame, SnakeGameState

import torch
from torch import nn

import sys

HIDDEN_SIZE1 = 20
HIDDEN_SIZE2 = 10


class SupervisedNN:
    class GameNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.stack = nn.Sequential(
                nn.Linear(9, HIDDEN_SIZE1),
                nn.ReLU(),
                nn.Linear(HIDDEN_SIZE1, HIDDEN_SIZE2),
                nn.ReLU(),
                nn.Linear(HIDDEN_SIZE2, 1)
            )

        def forward(self, x):
            return self.stack(x)

    def __init__(self, training: bool):
        self._training = training
        self.model = SupervisedNN.GameNN()

        if training:
            self.loss_fn = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)
            self.model.train()
            self.inputs = []
            self.outputs = []
            self.trained_samples = 0
        else:
            self.model.load_state_dict(torch.load("model.pth"))

    def train(self, state: SnakeGameState, direction: int) -> None:
        input = [state.headx,
                  state.heady,
                  state.foodx,
                  state.foody,
                  state.direction,
                  state.size,
                  state.onleft,
                  state.onstraight,
                  state.onright]
        self.inputs.append(input)
        self.outputs.append([direction])

        if len(self.inputs) > 1000:
            self.optimizer.zero_grad()
            inp = torch.tensor(self.inputs, dtype=torch.float)
            out = self.model(inp)

            outputs = torch.tensor(self.outputs, dtype=torch.float)
            loss = self.loss_fn(out, outputs)
            loss.backward()
            self.optimizer.step()

            self.trained_samples += len(self.inputs)
            print(f"loss: {loss.long():>7f} - {self.trained_samples: >10d}")

            self.inputs.clear()
            self.outputs.clear()

            torch.save(self.model.state_dict(), "model.pth")
        return

    def next(self, state: SnakeGameState) -> int:
        return 0