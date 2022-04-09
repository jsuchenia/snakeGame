# Snake ML model

Simple ML model to learn a silicon player how to play a Snake game. As a supervision there is a simple algorithm that is trying to find out a shortest path to a "food" and changing a direction one step before an obstacle. This is mostly to play with ML libraries and build a MlOps pipelines around that.


Goal is to score highest (longest) possible snake

## How to train

```python
./experiments.py --batch 10 --samples 30000 128 256
```

it will use 30k samples (moves) grouped in a batch of 10  with a network with two hidden layers 128 and 256 (plus 10 inputs and 4 outputs). After each batch NN will be recalculated. NN will calculate a probalility of a move in each direction and then the highest value will be used. A network is used to play one game every 100 samples to check a progres. The highest score over a whole training session is used as a training score

## How to visualise
For a visualisaion a pygame library is used.

### Play game using keyboard
To play you can use arrows and pause using a space

```python
./play.py play
```

### Supervision algorithm visualisation
To visualise superviion algorithm you can run

```python
./play.py playsimple
```

### To run a trained model you can 

```python
./play.py playAI models/experiment_batch10_samples30000_skip-256-512
```