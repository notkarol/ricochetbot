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

    def __init__(self, max_depth, include_black_robot, include_redirects, quadrant_path):
        # Handle Arguments
        self.__max_depth = max_depth
        self.__include_black_robot = include_black_robot
        self.__include_redirects = include_redirects
        self.__quadrant_path = quadrant_path

        # Initialize rest of class variables
        self.__markers = ricochet.markers()
        self.__colors = self.__markers[4:9]
        self.__shapes = ['o', '^', 's', 'h', 'p']
        self.__targets = np.zeros((17, 4), dtype=np.int64)
        self.__robots = np.zeros((4 + self.__include_black_robot, 2),  dtype=np.int64)
        self.__grid = np.zeros((16, 16), dtype=np.int64)
        self.__solution = None
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
            num_quadrants = len(self.__quadrants[color]) - (1 - self.__include_redirects)
            quadrant_index = np.random.randint(num_quadrants)
            quadrant = self.__quadrants[color][quadrant_index]

            # Populate targets, walls, and reflectors based on our need to rotate
            for name, x, y in quadrant:
                if self.__is_target(name):
                    x, y = np.matmul(np.array([x, y]), rot_matrix) + offset + 8
                    self.__grid[y, x] |= 1 << self.__markers.index(name)
                    self.__targets[target_index, :] = [self.__colors.index(name[0]),
                                                       self.__shapes.index(name[1]), x, y]
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
                    x, y = np.matmul(np.array([x, y]), rot_matrix) + offset + 8
                    if flip_walls:
                        name = name[0] + ('i' if name[1] == 'd' else 'i')
                    self.__grid[y, x] |= 1 << self.__markers.index(name)

        # Make sure targets are not in predictable order
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

    def done(self):
        return self.__turn >= len(self.__targets)

    def next(self):
        self.__turn += 1
        self.__solution = None

    def turn(self, turn):
        self.__turn = turn
        
    def solve(self, solver):
        color_i, shape_i, x, y = [int(x) for x in self.__targets[self.__turn]]
        self.__solution = ricochet.solve(self.__grid, self.__robots, color_i, x, y,
                                         self.__max_depth, solver)
        if self.__solution is not None:
            print("%2i" % len(self.__solution))
        else:
            print("N/A")

    def plot(self, save_name='board'):
        fig = plt.gcf()
        ax = fig.add_subplot(111)
        ax.set_facecolor("gainsboro")
        for side in ['top', 'bottom', 'right', 'left']:
            ax.spines[side].set_visible(False)
        plt.tick_params(axis='x', bottom='off')
        plt.tick_params(axis='y', left='off')
        fig.patch.set_facecolor('black')
        fig.set_size_inches(10, 10)
        plt.axis([0, 16, 16, 0])
        plt.xticks(np.arange(17), [])
        plt.yticks(np.arange(17), [])
        plt.grid()

        # Targets
        for i, (color_i, shape_i, x, y) in enumerate(self.__targets):
            alpha = 0.1
            name = self.__colors[color_i] + self.__shapes[shape_i]
            if i == self.__turn:
                alpha = 1
                plt.plot(8, 8, name, ms=64)
            plt.plot(x + 0.5, y + 0.5, name, alpha=alpha, ms=32, markeredgecolor='gray')

        # Robots
        for i, (x, y) in enumerate(self.__robots):
            plt.plot(x + 0.5, y + 0.5, self.__colors[i] + '*', ms=32, markeredgecolor='gray')

        # Walls
        for y in range(16):
            for x in range(16):
                if self.__grid[y, x] & 1 << self.__markers.index('wu'):
                    plt.plot([x, x + 1], [y, y], lw=4*(1+(y==0)), color='gray')    
                if self.__grid[y, x] & 1 << self.__markers.index('wd'):
                    plt.plot([x, x + 1], [y + 1, y + 1], lw=4*(1+(y==15)), color='gray')    
                if self.__grid[y, x] & 1 << self.__markers.index('wl'):
                    plt.plot([x, x], [y, y + 1], lw=4*(1+(x==0)), color='gray')
                if self.__grid[y, x] & 1 << self.__markers.index('wr'):
                    plt.plot([x + 1, x + 1], [y, y + 1], lw=4*(1+(x==15)), color='gray')

        # Redirects
        redirect_mask = ricochet.redirect_mask()
        for y in range(16):
            for x in range(16):
                if self.__grid[y, x] & redirect_mask:
                    for i, marker in enumerate(self.__markers):
                        if len(marker) >= 2 and marker[1] == 'd' and self.__grid[y, x] & 1 << i:
                            plt.plot([x, x + 1], [y, y+ 1], lw=4, color=marker[0], alpha=0.5)
                        if len(marker) >= 2 and marker[1] == 'i' and self.__grid[y, x] & 1 << i:
                            plt.plot([x, x + 1], [y + 1, y], lw=4, color=marker[0], alpha=0.5)

        # Path to solution
        if self.__solution:
            plt.text(8, 8,  str(len(self.__solution)), horizontalalignment='center',
                     verticalalignment='center', fontsize=32, color='lightgray')
            for robot_i, action_i, src_x, src_y, dst_x, dst_y in self.__solution:
                xs = [src_x + 0.5, dst_x + 0.5]
                ys = [src_y + 0.5, dst_y + 0.5]
                if src_x != dst_x and src_y != dst_y:
                    xs.insert(1, (dst_x if action_i % 2 else src_x) + 0.5)
                    ys.insert(1, (src_y if action_i % 2 else dst_y) + 0.5)
                for dx, dy in [(0.05, 0.05), (0, -0.1), (-0.1, 0.0), (0, 0.1)]:
                    xs[0] += dx
                    ys[0] += dy
                    if len(xs) == 3:
                        xs[1] += dx / 2
                        ys[1] += dy / 2
                    plt.plot(xs, ys, self.__colors[robot_i], lw=3, alpha=0.2)

        plt.savefig('%s.png' % save_name, bbox_inches='tight', dpi=100, facecolor='gainsboro')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=-1,
                        help="Seed to initialize board/robots/order with. Less than 0 is random")
    parser.add_argument("--max", type=int, default=12,
                        help="Maximum number of moves to try to solve in. More moves means that it"
                        "'s likely to take quadratically longer to solve board.")
    parser.add_argument("--redirect", action="store_true",
                        help="Alternate mode: possibly include board tiles that have redirects")
    parser.add_argument("--black", action="store_true",
                        help="Alternate mode: include a fifth robot")
    parser.add_argument("--path", type=str, default="config/quadrants.yaml",
                        help="Path to quadrants yaml in case you want to design your own boards.")
    parser.add_argument("--plot", action="store_true", help="Plot the board")
    parser.add_argument("--solver", type=int, default=1, help="Solver to use. 1 is depth-first."
                        "2 is breadth-first. 3 is bi-directional")
    args = parser.parse_args()

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)

    b = Board(args.max, args.black, args.redirect, args.path)
    counter = 0
    while not b.done():
        print("Solving %2i:" % counter, end=' ')
        if args.plot:
            b.plot('board_%02i_ready' % counter)
        b.solve(args.solver)
        if args.plot:
            b.plot('board_%02i_solved' % counter)
        b.next()
        counter += 1
