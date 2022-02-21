from game import SnakeGame, SnakeGameState

import torch
from torch import nn

import os


class SupervisedNN:
    class GameNN(nn.Module):
        def __init__(self, layers):
            super().__init__()
            elements = []
            for i in range(len(layers)):
                if i == 0:
                    elements.append(nn.Linear(9, layers[i]))
                else:
                    elements.append(nn.Linear(layers[i-1], layers[i]))
                elements.append(nn.ReLU())
            elements.append(nn.Linear(layers[-1], 3))

            self.stack = nn.Sequential(*elements)

        def forward(self, x):
            return self.stack(x)

    def __init__(self, training: bool, layers, prefix="model"):
        self._training = training
        self.model = SupervisedNN.GameNN(layers)
        suffix = '-'.join([str(layer) for layer in layers])
        self._modelpath = f"models/{prefix}-{suffix}"

        if self._modelpath and os.path.exists(self._modelpath):
            self.model.load_state_dict(torch.load(self._modelpath))

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.inputs = []
        self.outputs = []
        self.trained_samples = 0

    def getstatetorch(self, state:SnakeGameState):
        return [state.headx,
                 state.heady,
                 state.foodx,
                 state.foody,
                 state.direction,
                 state.size,
                 state.onleft,
                 state.onstraight,
                 state.onright]

    def train(self, state: SnakeGameState, direction: int) -> None:
        self.inputs.append(self.getstatetorch(state))
        output = [0, 0, 0]
        output[direction+1] = 1
        self.outputs.append(output)

        if len(self.inputs) >= 10000:
            inp = torch.tensor(self.inputs, dtype=torch.float)
            out = self.model(inp)

            outputs = torch.tensor(self.outputs, dtype=torch.float)
            loss = self.loss_fn(out, outputs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.trained_samples += len(self.inputs)
            self.inputs.clear()
            self.outputs.clear()

            if self._modelpath:
                torch.save(self.model.state_dict(), self._modelpath)
        return

    def next(self, state: SnakeGameState) -> int:
        input = torch.tensor([self.getstatetorch(state)], dtype=torch.float)
        output = self.model(input)
        o = int(output[0].argmax(0)) - 1
        return o
