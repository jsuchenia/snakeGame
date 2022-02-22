#!/bin/bash

. venv/bin/activate

#Compare net size
python experiments.py --batch 10 --samples 10000 10 20
python experiments.py --batch 10 --samples 10000 20 40
python experiments.py --batch 10 --samples 10000 48 96
python experiments.py --batch 10 --samples 10000 96 48
python experiments.py --batch 10 --samples 10000 128 256
python experiments.py --batch 10 --samples 10000 256 128


#Compare batch size (for only 48 96)
python experiments.py --batch 10 --samples 10000 48 96
python experiments.py --batch 20 --samples 10000 48 96
python experiments.py --batch 100 --samples 10000 48 96


#Compare long simulation
python experiments.py --batch 10 --samples 30000 48 96
python experiments.py --batch 10 --samples 30000 128 256

