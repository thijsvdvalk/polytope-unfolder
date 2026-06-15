# 4-Polytope Unfolder for a Heuristic Algorithms Evaluation

This repository contains code that implements:
1. Functionality to randomly generate convex 4-polytopes. See `src/polytope_core/builder.py`
2. Implementations of various heuristic algorithms that choose unfoldings. See `src/heur_algo_imps/*`
3. A polytope unfolder along with overlap detection functionality. See `src/polytope.py`

## Setup environment

All the dependencies are listed in `requirements.txt`. The repository is developend and tested on a Linux machine running `python 3.14.4`, however it should work fine for other OS's and probably for most other python versions >=3.0.0 as well.

## Generate the dataset

To reproduce the dataset that was used for evaluation of the heuristic algorithms, simply just run the following command from the project root:

```bash
python3 scripts/generate_dataset.py
```

## Run the heuristic algorithms

To reproduce the runs of the heuristic algorithms, simply tjust run the following command from the project root, given that the dataset has already been implemented:


```bash
python3 scripts/run_heuros.py
```

