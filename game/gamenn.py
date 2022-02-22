from game.game import SnakeGame, SnakeGameState

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
                    elements.append(nn.Linear(10, layers[i]))
                else:
                    elements.append(nn.Linear(layers[i-1], layers[i]))
                elements.append(nn.ReLU())
            elements.append(nn.Linear(layers[-1], 3))

            self.stack = nn.Sequential(*elements)

        def forward(self, x):
            return self.stack(x)

    def __init__(self, training: bool, layers, modelfile, batchsize=10):
        self._training = training
        self.model = SupervisedNN.GameNN(layers)
        self.batchsize = batchsize

        if modelfile:
            self._modelpath = modelfile

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        if self._modelpath and os.path.exists(self._modelpath):
            self.model.load_state_dict(torch.load(self._modelpath))

        self.inputs = []
        self.outputs = []
        self.trained_samples = 0

        if training:
            self.model.train()
        else:
            self.model.eval()

    def getstatetorch(self, state:SnakeGameState):
        return [state.headx,
                 state.heady,
                 state.foodx,
                 state.foody,
                 state.foodx-state.headx,
                 state.foody-state.heady,
                 state.direction,
                 state.onleft,
                 state.onstraight,
                 state.onright]

    def trainmode(self):
        self.model.train(True)

    def usemode(self):
        self.model.train(False)

    def train(self, state: SnakeGameState, direction: int) -> None:
        self.inputs.append(self.getstatetorch(state))
        output = [0, 0, 0]
        output[direction + 1] = 1
        self.outputs.append(output)

        if len(self.inputs) >= self.batchsize:
            self.trained_samples += len(self.inputs)
            self.optimizer.zero_grad()

            inputstensor = torch.tensor(self.inputs, dtype=torch.float)
            outputstensor = torch.tensor(self.outputs, dtype=torch.float)

            result = self.model(inputstensor)
            loss = self.loss_fn(result, outputstensor)

            loss.backward()
            self.optimizer.step()

            self.inputs.clear()
            self.outputs.clear()

        return

    def save(self):
        if self._modelpath:
            torch.save(self.model.state_dict(), self._modelpath)

    def next(self, state: SnakeGameState) -> int:
        input = torch.tensor([self.getstatetorch(state)], dtype=torch.float)
        with torch.no_grad():
            output = self.model(input)
            return int(output[0].argmax(0).item()) - 1
