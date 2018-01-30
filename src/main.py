#!/usr/bin/python3

from copy import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import yaml

MARKERS = ['wu', 'wd', 'wl', 'wr',
           'b', 'g', 'r', 'y', 
           'bh', 'bo', 'bs', 'bv',
           'gh', 'go', 'gs', 'gv',
           'rh', 'ro', 'rs', 'rv',
           'yh', 'yo', 'ys', 'yv',
           'kp', 'kx']
MAPPING = {marker: i for i, marker in enumerate(MARKERS)}


def load_quadrants(path="../config/quadrants.yaml"):
    """ 
    Loads the quadrants yaml and automatically populates center/edge walls
    Returns the quadrants as a dictionary by color
    """
    with open(path) as f:
        quadrants = yaml.load(f)

    for color in sorted(quadrants):
        for quadrant in quadrants[color]:
            quadrant['hwalls'].append([0, 1])
            quadrant['vwalls'].append([1, 0])
            quadrant['hwalls'].extend([[i, 8] for i in range(8)])
            quadrant['vwalls'].extend([[8, i] for i in range(8)])
    return quadrants


def plot_board(board, name=0):
    """
    Plots a given board for verification purposes
    """
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.axis([0, 16, 16, 0])
    plt.xticks(np.arange(17))
    plt.yticks(np.arange(17))
    plt.grid()

    current_target = board['order'][board['turn']]
    for target in board['targets']:
        x, y = board['targets'][target]
        alpha = 1 if current_target == target else 0.33
        plt.plot(x + 0.5, y + 0.5, target, alpha=alpha, ms=32)

    for robot in board['robots']:
        x, y = board['robots'][robot]
        plt.plot(x + 0.5, y + 0.5, robot + '*', ms=24)

    for x, y in board['hwalls']:
        plt.plot([x, x + 1], [y, y], lw=4, color='gray')

    for x, y in board['vwalls']:
        plt.plot([x, x], [y, y + 1], lw=4, color='gray')

    plt.plot(8, 8, current_target, ms=64)
        
    plt.savefig('board_%02i.png' % name, bbox_inches='tight', dpi=100)
    plt.close()
    
    
def create_board(quadrants, include_black_robot=False):
    """ 
    Prepare a board from a random select of each color's quadrant
    """

    board = {'targets': {}, # each target and its location on board
             'hwalls': [], # locations of horizontal walls
             'vwalls': [], # locations of vertical walls
             'robots': {}, # each robot and its location on board
             'order': [], # shuffled target names for order to solve
             'turn': 0, # current turn number
             'moves': [], # moves to solve each subsequent target in order
             'grid': np.zeros((16, 16), dtype=np.int64)} # targets/walls/robots for calculations
    # Prepare a random order of colors
    colors = ['red', 'blue', 'green', 'yellow']
    random.shuffle(colors)

    # Prepare board
    for color_i, color in enumerate(colors):

        # Figure out the angle we're rotating by and the necessary offsets
        # This is a lazy approach to rotation
        rot = color_i * 90 / 180 * np.pi
        flip_walls = color_i in [1, 3]
        offset = np.array([color_i in [2, 3], color_i in [1, 2]], dtype=np.int64) * -1
        rot_matrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]],
                              dtype=np.int64)

        # Pick a quadrant
        quadrant_index = np.random.randint(len(quadrants[color]))
        quadrant = quadrants[color][quadrant_index]

        # Get new positions for each tile
        for target in sorted(quadrant['targets']):
            xy = quadrant['targets'][target]
            board['targets'][target] = np.matmul(xy, rot_matrix) + offset + 8
            if target[-1] != '.':
                board['order'].append(target)

        # Get new positions for each wall
        wall_names = ['hwalls', 'vwalls']
        for wall_i, wall in enumerate(wall_names):
            for x, y in quadrant[wall]:
                new_wall_i = (wall_i + flip_walls) % 2
                target = wall_names[new_wall_i]
                new_xy = np.matmul([x, y], rot_matrix) + 8
                new_xy[new_wall_i] += offset[new_wall_i]
                board[target].append(new_xy)

    # Prepare grid of board
    board['grid'][7:9, 7:9] = MAPPING['kx'] # do not include center tiles
    for target in board['targets']:
        x, y = board['targets'][target]
        board['grid'][y, x] |= 1 << MAPPING[target]
    for x, y in board['hwalls']:
        if y < 16:
            board['grid'][y, x] |= 1 << MAPPING['wu']
        if y > 0:
            board['grid'][y-1, x] |= 1 << MAPPING['wd']
    for x, y in board['vwalls']:
        if x < 16:
            board['grid'][y, x] |= 1 << MAPPING['wl']
        if x > 0:
            board['grid'][y, x-1] |= 1 << MAPPING['wr']

    # Populate robots in random positions not on a target
    if include_black_robot:
        colors.append('k')
    for color in colors:
        x, y = 8, 8
        while board['grid'][y, x] >> 4:
            x, y = np.random.randint(16, size=2)
        robot = color[0]
        xy = np.array([x, y], dtype=np.int64)
        board['grid'][y][x] |= 1 << MAPPING[robot]
        board['robots'][robot] = xy

    # Shuffle order to simulate picking random targets
    random.shuffle(board['order'])
        
    return board


def descend(grid, robots, target, move=None, counter=0):
    """
    Try recursively moving each robot in one of their directions to get to the target
    """
    if counter >= 8:
        return None

    return []


def solve(board):
    """
    Calls descend recursively until a solution is found
    """
    grid = board['grid'] & 255
    current_target = board['order'][board['turn']]
    robots = copy(board['robots'])
    target_x, target_y = board['targets'][current_target]
    target_color = current_target[0]
    solution = descend(grid, robots, (target_x, target_y, target_color))
    print(len(solution))
    board['turn'] += 1


if __name__ == "__main__":
    # Play a game on the given board
    quadrants = load_quadrants()
    board = create_board(quadrants)
    plot_board(board)
    while board['turn'] < len(board['order']):
        solve(board)
