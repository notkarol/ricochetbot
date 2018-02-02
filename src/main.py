#!/usr/bin/python3

import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import random
import yaml

class Piece:
    def __init__(self, name, x, y):
        self.name = name
        self.x, self.y = (x, y)
        self.xy = np.array([x, y], dtype=np.int)

    def __str__(self):
        return "(%s,%i,%i)" % (self.name, self.x, self.y)

    def __repr__(self):
        return "(%s,%i,%i)" % (self.name, self.x, self.y)

    def rename(self, name):
        self.name = name

    def move(self, x, y):
        self.x = int(x)
        self.y = int(y)
        self.xy[:] = (x, y)

    def is_wall(self):
        return len(self.name) == 2 and self.name[0] == 'w'

    def is_robot(self):
        return len(self.name) == 1

    def is_target(self):
        return len(self.name) == 2 and self.name[0] in 'bgryk' and self.name[1] in 'hosvp'

    def is_reflector(self):
        return len(self.name) == 2 and self.name[1] in '\\/'


class Board:

    def __init__(self):
        # Things that should be arguments
        self.__include_black_robot = False
        self.__quadrant_path = "../config/quadrants.yaml"
        self.__max_depth = 5

        # Initialize rest of program
        self.__colors = ['red', 'blue', 'green', 'yellow']
        self.__targets = []
        self.__walls = []
        self.__robots = []
        self.__turn = 0
        self.__grid = np.zeros((16, 16), dtype=np.int64)
        self.__grid_mask = 2**17 - 1 # first 17 pieces are walls or robots
        self.__piece_names = ['wu', 'wd', 'wl', 'wr',
                              'b', 'g', 'r', 'y', 'k',
                              'b\\', 'b/', 'g\\', 'g/', 'r\\', 'r/', 'y\\', 'y/',
                              'bh', 'bo', 'bs', 'bv', 'gh', 'go', 'gs', 'gv',
                              'rh', 'ro', 'rs', 'rv', 'yh', 'yo', 'ys', 'yv', 'kp',
                              'kx', 'wv', 'wh']
        self.__piece_bits = {name: index for index, name in enumerate(self.__piece_names)}

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


    def __create_board(self):
        """
        Prepare a board from a random select of each color's quadrant
        """
        wall_names = ['wh', 'wv']

        random.shuffle(self.__colors)
        for color_i, color in enumerate(self.__colors):

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
            for item in quadrant:
                piece = Piece(*item)

                if piece.is_target():
                    new_xy = np.matmul(piece.xy, rot_matrix) + offset + 8
                    piece.move(*new_xy)
                    self.__targets.append(piece)

                elif piece.is_wall():
                    wall_i = wall_names.index(piece.name)
                    new_wall_i = (wall_i + flip_walls) % 2
                    new_xy = np.matmul(piece.xy, rot_matrix) + 8
                    new_xy[new_wall_i] += offset[new_wall_i]
                    if flip_walls:
                        piece.rename(wall_names[new_wall_i])
                    piece.move(*new_xy)
                    self.__walls.append(piece)

                elif piece.is_reflector():
                    raise NotImplementedError("Reflectors are not yet implemented")

        # Shuffle order of targets to simulate picking random targets
        random.shuffle(self.__targets)

        # Prepare grid of board
        self.__grid[7:9, 7:9] = self.__piece_bits['kx'] # do not let robots populate center
        for target in self.__targets:
            self.__grid[target.y, target.x] |= 1 << self.__piece_bits[target.name]
        for wall in self.__walls:
            try:
                if wall.name == 'wh':
                    if wall.y < 16:
                        self.__grid[wall.y, wall.x] |= 1 << self.__piece_bits['wu']
                    if wall.y > 0:
                        self.__grid[wall.y - 1, wall.x] |= 1 << self.__piece_bits['wd']
                if wall.name == 'wv':
                    if wall.x < 16:
                        self.__grid[wall.y, wall.x] |= 1 << self.__piece_bits['wl']
                    if wall.x > 0:
                        self.__grid[wall.y, wall.x - 1] |= 1 << self.__piece_bits['wr']
            except:
                pass


    def __populate_robots(self):

        # Populate robots in random positions not on a target
        if self.__include_black_robot:
            self.__colors.append('k')
        for color in self.__colors:

            # Find a location for the robot
            x, y = 8, 8
            while self.__grid[y, x] >> 4:
                x, y = np.random.randint(16, size=2)

            # Prepare new location
            robot = Piece(color[0], x, y)
            self.__robots.append(robot)
            self.__grid[y, x] |= 1 << self.__piece_bits[robot.name]


    def descend(self, grid, robots, target_color, target_x, target_y, move=None, counter=0):
        """
        Try recursively moving each robot in one of their directions to get to the target
        """
        if counter >= self.__max_depth:
            return None

        # If we were asked to make a move, do it
        if move is not None:
            robot_color, src_x, src_y, dst_x, dst_y = move

            # If we didn't move, return None
            if src_x == dst_x and src_y == dst_y:
                return None
            
            # if we've found the solution return it
            if robot_color == target_color and target_x == dst_x and target_y == dst_y:
                return [move]
            
            # Move the robot by reseting the grid and updating the robots variable
            robot_bit = 1 << self.__piece_bits[robot_color]
            grid[src_y, src_x] &= robot_bit ^ 511
            grid[dst_y, src_x] |= robot_bit
            for robot in robots:
                if robot.name == robot_color:
                    robot.move(dst_x, dst_y)
                    
        # Find possible paths recursively
        outs = []
        for robot in robots:
            for i, func in enumerate([self.get_left_destination, self.get_down_destination,
                                      self.get_right_destination, self.get_up_destination]):
                dst_x, dst_y = func(grid, robot.x, robot.y)

                next_move = [robot.name, robot.x, robot.y, dst_x, dst_y]
                out = self.descend(grid.copy(), deepcopy(robots), target_color,
                                   target_x, target_y, next_move, counter + 1)
                if out is not None:
                    outs.append(out)

        # Return the shortest path that we find
        min_out_n = self.__max_depth
        min_out_val = None
        for out in outs:
            if len(out) < min_out_n:
                min_out_n = len(out)
                min_out_val = ([move] if move else []) + out
        return min_out_val


    def solve(self):
        """
        Calls descend recursively until a solution is found
        """
        grid = self.__grid & 511 # only store walls and robots
        target = self.__targets[self.__turn]
        target_color = target.name[0]
        robots = deepcopy(self.__robots)
        solution = self.descend(grid, robots, target_color, target.x, target.y)
        print(solution)
        self.__turn += 1


    def get_current_target(self):
        return self.__targets[self.__turn]


    def get_left_destination(self, grid, x, y):
        wall = 1 << self.__piece_bits['wr']
        while x > 0 and grid[y, x - 1] & wall == 0 and grid[y, x - 1] >> 4 == 0:
            x -= 1
        return x, y 


    def get_right_destination(self, grid, x, y):
        wall = 1 << self.__piece_bits['wl']
        while x < 15 and grid[y, x + 1] & wall == 0 and grid[y, x + 1] >> 4 == 0:
            x += 1
        return x, y 


    def get_up_destination(self, grid, x, y):
        wall = 1 << self.__piece_bits['wd']
        while y > 0 and grid[y - 1, x] & wall == 0 and grid[y - 1, x] >> 4 == 0:
            y -= 1
        return x, y 


    def get_down_destination(self, grid, x, y):
        wall = 1 << self.__piece_bits['wu']
        while y < 15 and grid[y + 1, x] & wall == 0 and grid[y + 1, x] >> 4 == 0:
            y += 1
        return x, y 


    def plot(self, name='board'):
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        plt.axis([0, 16, 16, 0])
        plt.xticks(np.arange(17))
        plt.yticks(np.arange(17))
        plt.grid()

        current_target = self.get_current_target()
        for target in self.__targets:
            alpha = 1 if current_target.name == target.name else 0.33
            plt.plot(target.x + 0.5, target.y + 0.5, target.name, alpha=alpha, ms=32)

        for robot in self.__robots:
            plt.plot(robot.x + 0.5, robot.y + 0.5, robot.name + '*', ms=24)

        for wall in self.__walls:
            if wall.name == 'wh':
                plt.plot([wall.x, wall.x + 1], [wall.y, wall.y], lw=4, color='gray')
            elif wall.name == 'wv':
                plt.plot([wall.x, wall.x], [wall.y, wall.y + 1], lw=4, color='gray')
            else:
                raise ValueError("Not a wall [%s]" % wall.name)
        plt.plot(8, 8, current_target.name, ms=64)

        plt.savefig('%s.png' % name, bbox_inches='tight', dpi=100)
        plt.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args()

    # Seed
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Play a game on the given board
    b = Board()
    b.plot('board0')
    b.solve()
    b.plot('board1')
