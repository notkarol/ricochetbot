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

    def __init__(self, quadrants, include_black_robot):
        self.__include_black_robot = include_black_robot

        # Initialize rest of class variables
        self.__markers = ["wu", "wd", "wl", "wr", "b", "g", "r", "y", "k",
                          "bi", "bd", "gi", "gd", "ri", "rd", "yi", "yd",
                          "bo", "b^", "bs", "bh", "go", "g^", "gs", "gh",
                          "ro", "r^", "rs", "rh", "yo", "y^", "ys", "yh",
                          "kp", ""]
        self.__colors = self.__markers[4:9]
        self.__shapes = ['o', '^', 's', 'h', 'p']
        self.__target_list = np.zeros((17, 4), dtype=np.int8)
        self.__robot_list = np.zeros((4 + self.__include_black_robot, 2),  dtype=np.int8)
        self.__wall_grids = {wall: np.zeros((16, 16), dtype=np.float) for wall in 'hv'}
        self.__robot_grid = np.zeros((16, 16), dtype=np.float)
        self.__target_grid = np.zeros((16, 16), dtype=np.float)

        # Prepare board
        self.__construct(quadrants)
        self.__add_robots()

    def __is_target(self, name):
        return len(name) == 2 and name[0] in self.__colors and name[1] in self.__shapes

    def __is_wall(self, name):
        return len(name) == 2 and name[0] == 'w'

    def __can_place_at(self, x, y):
        # Do not place in center or on top of another robot
        if x == 7 or x == 8 or y == 7 or y == 8 or self.__robot_grid[y, x]:
            return False
        
        # Do not place on top of target. We can do some numpy here but keep this simple.
        for _, _, target_x, target_y in self.__target_list:
            if target_x == x and target_y == y:
                return False
        return True
        
    def __construct(self, quadrants):
        """
        Prepare a board from a random select of each color's quadrant
        """

        # Names of walls, useful for eventually flipping horizontal/vertical when we rotate
        wall_names = ['wh', 'wv']
        target_index = 0        
        color_names = ['red', 'blue', 'green', 'yellow']
        random.shuffle(color_names)

        # Loop through each color to populate each quadrants
        for color_i, color in enumerate(color_names):

            # Figure out the angle we're rotating by and the necessary offsets
            # This is a lazy approach to rotation
            rot = color_i * 90 / 180 * np.pi
            flip_walls = color_i in [1, 3]
            offset = np.array([color_i in [2, 3], color_i in [1, 2]], dtype=np.int64) * -1
            rot_matrix = np.array([[np.cos(rot), -np.sin(rot)],
                                   [np.sin(rot), np.cos(rot)]], dtype=np.int64)

            # Pick a quadrant
            quadrant_index = np.random.randint(len(quadrants[color]))
            quadrant = quadrants[color][quadrant_index]

            # Populate targets, walls, and reflectors based on our need to rotate
            for name, x, y in quadrant:
                
                if self.__is_target(name):
                    color = self.__colors.index(name[0])
                    shape = self.__shapes.index(name[1])
                    dst_x, dst_y = np.matmul(np.array([x, y]), rot_matrix) + offset + 8
                    self.__target_list[target_index, :] = (color, shape, dst_x, dst_y)
                    target_index += 1

                elif self.__is_wall(name):
                    wall_i = (wall_names.index(name) + flip_walls) % 2
                    name = wall_names[wall_i] if flip_walls else name
                    xy = np.matmul(np.array([x, y]), rot_matrix) + 8
                    xy[wall_i] += offset[wall_i]
                    dst_x, dst_y = xy
                    self.__wall_grids[name[1]][dst_y, dst_x] = 1
                            
        # Make sure targets are not in predictable order
        np.random.shuffle(self.__target_list)

    def __add_robots(self):

        # Populate robots in random positions not on a target
        for i in range(len(self.__robot_list)):
            
            # Find a location for the robot
            x, y = 8, 8
            while not self.__can_place_at(x, y):
                x, y = np.random.randint(16, size=2)

            # Prepare new location
            self.__robot_list[i, :] = (x, y)

    def solve(self, solver, turn):
        color_i, shape_i, x, y = [int(x) for x in self.__target_list[turn]]
        move_list = ricochet.solve(self.__robot_grid, self.__target_grid,
                                   self.__wall_grids['h'], self.__wall_grids['v'],
                                   self.__robot_list, color_i, x, y, self.__max_depth, solver)
        if move_list is not None:
            print("%2i" % len(move_list), move_list)
        else:
            print("N/A")
        return move_list
    
    def plot(self, save_name='board', turn=0, move_list=None):
        fig = plt.gcf()
        ax = fig.add_subplot(111)
        ax.set_facecolor("gainsboro")
        for side in ['top', 'bottom', 'right', 'left']:
            ax.spines[side].set_visible(False)
        plt.tick_params(axis='x', bottom='off')
        plt.tick_params(axis='y', left='off')
        fig.patch.set_facecolor('black')
        fig.set_size_inches(10, 10)
        plt.axis([-0.05, 16.05, 16.05, -0.05])
        plt.xticks(np.arange(17), [])
        plt.yticks(np.arange(17), [])
        plt.grid()

        # Targets
        for i, (color_i, shape_i, x, y) in enumerate(self.__target_list):
            alpha = 0.0625
            name = self.__colors[color_i] + self.__shapes[shape_i]
            if i == turn:
                alpha = 1
                plt.plot(8, 8, name, ms=64, alpha=0.5)
            plt.plot(x + 0.5, y + 0.5, name, alpha=alpha, ms=32, markeredgecolor='gray')

        # Robots
        for i, (x, y) in enumerate(self.__robot_list):
            plt.plot(x + 0.5, y + 0.5, self.__colors[i] + '*', ms=32, markeredgecolor='gray')

        # Walls
        for y in range(16):
            plt.plot([y, y + 1], [0, 0], lw=4, color='gray')    
            plt.plot([y, y + 1], [16, 16], lw=4, color='gray')    
            plt.plot([0, 0], [y, y + 1], lw=4, color='gray')
            plt.plot([16, 16], [y, y + 1], lw=4, color='gray')
            for x in range(16):
                if self.__wall_grids['h'][y, x]:
                    plt.plot([x, x + 1], [y, y], lw=4*(1+(y==0)), color='gray')    
                if self.__wall_grids['v'][y, x]:
                    plt.plot([x, x], [y, y + 1], lw=4*(1+(x==0)), color='gray')

        # Path to solution
        if move_list:
            plt.text(8, 8,  str(len(move_list)), horizontalalignment='center',
                     verticalalignment='center', fontsize=32, color='lightgray')
            for robot_i, action_i, src_x, src_y, dst_x, dst_y in move_list:
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

def load_quadrants(path):
    """
    Loads the quadrants yaml and adds inner and outer walls to each quadrant
    Returns the quadrants as a dictionary by color
    """
    with open(path) as f:
        quadrants = yaml.load(f)

    # clear redirects since we do not want to deal with them yet
    for color in quadrants:
        del quadrants[color][-1]

    for color in quadrants:
        for quadrant in quadrants[color]:
            quadrant.append(['wh', 0, 1])
            quadrant.append(['wv', 1, 0])

    return quadrants
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=-1,
                        help="Seed to initialize board/robots/order with. Less than 0 is random")
    parser.add_argument("--max", type=int, default=12,
                        help="Maximum number of moves to try to solve in. More moves means that it"
                        "'s likely to take quadratically longer to solve board.")
    parser.add_argument("--black", action="store_true",
                        help="Alternate mode: include a fifth robot")
    parser.add_argument("--path", type=str, default="config/quadrants.yaml",
                        help="Path to quadrants yaml in case you want to design your own boards.")
    parser.add_argument("--plot", action="store_true", help="Plot the board")
    parser.add_argument("--solver", type=int, default=1, help="Solver to use. 1 is depth-first."
                        "2 is breadth-first. 3 is bi-directional")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    quadrants = load_quadrants(args.path)
    
    b = Board(quadrants, args.black)
    b.plot('board')

    counter = 0
    for turn in range(17):
        print("Solving %2i:" % counter, end=' ')
        if args.plot:
            b.plot('board_%02i_ready' % counter, turn)
        move_list = None #b.solve(args.solver, turn)
        if args.plot:
            b.plot('board_%02i_solved' % counter, turn, move_list)
        counter += 1
