# ricochetbot

An unofficial Ricochet Robots solver. This repository serves as a test various heuristics for solving the game.

![alt text](https://raw.githubusercontent.com/notkarol/ricochetbot/master/images/example.png)

## Prerequisites

* python3
* python3-matplotlib
* python3-numpy
* python3-yaml

## Instructions

1. Compile the code.
```bash
./setup.py build_ext --inplace
```
2. Generate a board, robots, and then solve it for a random order of targets.
```bash
./main.py
```
3. View results. The ready image is the state of the board before the solved image solves it.
```bash
eog board*png
```

## Goals

1. Generate a random board of all possible board, target, robot combinations.
2. Find the shortest route for a given board, target, and robots.
3. Analyze attributes of shortest routes.
4. Use computer vision to solve games live from a camera feed.
5. Evaluate the extent that reinforcement learning can solve this issue.
