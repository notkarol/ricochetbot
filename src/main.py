#!/usr/bin/python3

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import yaml
import ricochet

class Board:

    def __init__(self, max_depth=8, include_black_robot=True,
                 quadrant_path="../config/quadrants.yaml"):
        # Handle Arguments
        self.__include_black_robot = include_black_robot
        self.__quadrant_path = quadrant_path
        self.__max_depth = max_depth

        # Initialize rest of class variables
        self.__markers = ricochet.markers()
        self.__colors = self.__markers[4:9]
        self.__shapes = ['o', 'v', 's', 'h', 'p']
        self.__targets = np.zeros((17, 4), dtype=np.int64)
        self.__robots = np.zeros((4 + self.__include_black_robot, 2),  dtype=np.int64)
        self.__grid = np.zeros((16, 16), dtype=np.int64)
        self.__turn = 0

        # Prepare board
        self.__load_quadrants()
        self.__create_board()
        self.__populate_robots()

    def __load_quadrants(self):
        """
        Loads the quadrants yaml and adds inner and outer walls to each quadrant
        Returns the quadrants as a dictionary by color
        """
        with open(self.__quadrant_path) as f:
            self.__quadrants = yaml.load(f)

        for color in self.__quadrants:
            for quadrant in self.__quadrants[color]:
                quadrant.append(['wh', 0, 1])
                quadrant.append(['wv', 1, 0])
                quadrant.extend([['wh', i, 8] for i in range(8)])
                quadrant.extend([['wv', 8, i] for i in range(8)])

    def __is_robot(self, name):
        return len(name) == 1 and name[0] in self.__colors

    def __is_target(self, name):
        return len(name) == 2 and name[0] in self.__colors and name[1] in self.__shapes

    def __is_wall(self, name):
        return len(name) == 2 and name[0] == 'w'

    def __is_redirect(self, name):
        return len(name) == 2 and name[0] in self.__colors and name[1] in 'id'
                
    def __create_board(self):
        """
        Prepare a board from a random select of each color's quadrant
        """

        # Mark center 4 tiles as unavailable
        self.__grid[7:9, 7:9] = 1 << self.__markers.index("")

        # Names of walls, useful for eventually flipping horizontal/vertical when we rotate
        wall_names = ['wh', 'wv']

        # Keep track of the index in target we're currently on
        target_index = 0        

        # Go through each color in a random order and add it to the board
        color_names = ['red', 'blue', 'green', 'yellow']
        random.shuffle(color_names)
        for color_i, color in enumerate(color_names):

            # Figure out the angle we're rotating by and the necessary offsets
            # This is a lazy approach to rotation
            rot = color_i * 90 / 180 * np.pi
            flip_walls = color_i in [1, 3]
            offset = np.array([color_i in [2, 3], color_i in [1, 2]], dtype=np.int64) * -1
            rot_matrix = np.array([[np.cos(rot), -np.sin(rot)],
                                   [np.sin(rot), np.cos(rot)]], dtype=np.int64)

            # Pick a quadrant
            quadrant_index = np.random.randint(len(self.__quadrants[color]))
            quadrant = self.__quadrants[color][quadrant_index]

            # Populate targets, walls, and reflectors based on our need to rotate
            for name, x, y in quadrant:
                if self.__is_target(name):
                    x, y = np.matmul(np.array([x, y]), rot_matrix) + offset + 8
                    self.__grid[y, x] |= 1 << self.__markers.index(name)
                    self.__targets[target_index, :] = [self.__colors.index(name[0]),
                                                       self.__shapes.index(name[1]),
                                                       x, y]
                    target_index += 1
                elif self.__is_wall(name):
                    wall_i = (wall_names.index(name) + flip_walls) % 2
                    name = wall_names[wall_i] if flip_walls else name
                    xy = np.matmul(np.array([x, y]), rot_matrix) + 8
                    xy[wall_i] += offset[wall_i]
                    x, y = xy
                    if name == 'wh':
                        if y < 16:
                            self.__grid[y, x] |= 1 << self.__markers.index('wu')
                        if y > 0:
                            self.__grid[y - 1, x] |= 1 << self.__markers.index('wd')
                    if name == 'wv':
                        if x < 16:
                            self.__grid[y, x] |= 1 << self.__markers.index('wl')
                        if x > 0:
                            self.__grid[y, x - 1] |= 1 << self.__markers.index('wr')
                elif self.__is_redirect(name):
                    raise NotImplementedError("Reflectors are not yet implemented")
        np.random.shuffle(self.__targets)

    def __populate_robots(self):

        # Populate robots in random positions not on a target
        for i in range(len(self.__robots)):
            
            # Find a location for the robot
            x, y = 8, 8
            while self.__grid[y, x] >> 4:
                x, y = np.random.randint(16, size=2)

            # Prepare new location
            self.__robots[i, :] = (x, y)
            self.__grid[y, x] |= 1 << self.__markers.index(self.__colors[i])

    def turn(self):
        return self.__turn

    def done(self):
        return self.__turn >= len(self.__targets)

    def solve(self):
        color_i, shape_i, x, y = [int(x) for x in self.__targets[self.__turn]]
        solution = ricochet.solve(self.__grid, self.__robots, color_i, x, y, self.__max_depth)
        self.__turn += 1

    def plot(self, save_name='board'):
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        plt.axis([0, 16, 16, 0])
        plt.xticks(np.arange(17))
        plt.yticks(np.arange(17))
        plt.grid()

        # Targets
        for i, (color_i, shape_i, x, y) in enumerate(self.__targets):
            alpha = 0.33
            name = self.__colors[color_i] + self.__shapes[shape_i]
            if i == self.__turn:
                alpha = 1
                plt.plot(8, 8, name, ms=64)
            plt.plot(x + 0.5, y + 0.5, name, alpha=alpha, ms=32)

        # Robots
        for i, (x, y) in enumerate(self.__robots):
            plt.plot(x + 0.5, y + 0.5, self.__colors[i] + '*', ms=24)

        # Walls
        for y in range(16):
            for x in range(16):
                if self.__grid[y, x] & 1 << 0: # up
                    plt.plot([x, x + 1], [y, y], lw=4, color='gray')    
                if self.__grid[y, x] & 1 << 1: # down
                    plt.plot([x, x + 1], [y + 1, y + 1], lw=4, color='gray')    
                if self.__grid[y, x] & 1 << 2: # left
                    plt.plot([x, x], [y, y + 1], lw=4, color='gray')
                if self.__grid[y, x] & 1 << 3: # right
                    plt.plot([x + 1, x + 1], [y, y + 1], lw=4, color='gray')

        plt.savefig('%s.png' % save_name, bbox_inches='tight', dpi=100)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--max", type=int, default=6)
    args = parser.parse_args()

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)

    b = Board(args.max)
    b.plot('board00')
    counter = 0
    while not b.done():
        counter += 1
        b.solve()
        b.plot('board%02i' % counter)

