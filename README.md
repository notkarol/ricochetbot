# ricochetbot

An unofficial Ricochet Robots solver. This repository serves as a test various heuristics for solving the game.

## Goals

1. Simulate all possible board combinations.
2. Find the shortest route for a given board, target, and robots.
3. Analyze attributes of shortest routes.
4. Use computer vision to solve games live from a camera feed.

## Prerequisites

* python3
* matplotlib
* numpy
* yaml

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

## Example

![alt text](https://raw.githubusercontent.com/notkarol/ricochetbot/master/images/example.png)

