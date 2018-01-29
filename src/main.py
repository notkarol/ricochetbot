#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
import yaml

def load_quadrants(path="quadrants.yaml"):
    """ 
    Loads the quadrants yaml and automatically populates center/edge walls
    Returns the quadrants as a dictionary by color
    """
    with open("quadrants.yaml") as f:
        quadrants = yaml.load(f)

    for color in quadrants:
        for quadrant in quadrants[color]:
            quadrant['tiles']['%sx' % color[0]] = [0, 0]
            quadrant['hwalls'].append([0, 1])
            quadrant['vwalls'].append([1, 0])
            for i in range(8):
                quadrant['hwalls'].append([i, 8])
                quadrant['vwalls'].append([8, i])
    return quadrants


def plot_marker(x, y, marker, ms=24):
    """
    Plots the given marker at the given location
    """
    plt.plot(x + 0.5, y + 0.5, marker, ms=ms)


def plot_line(xs, ys, color='k', lw=5):
    """ 
    Draws a line at the given location
    """
    plt.plot(xs, ys, color, lw=lw)


def plot_board(board, name=0):
    """
    Plots a given board for verification purposes
    """
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.axis([-9, 9, 9, -9])
    plt.xticks(np.arange(-8, 9))
    plt.yticks(np.arange(-8, 9))
    plt.grid()

    for tile in board['tiles']:
        x, y = board['tiles'][tile]
        plot_marker(x, y, tile)

    for x, y in board['hwalls']:
        plot_line([x, x + 1], [y, y])

    for x, y in board['vwalls']:
        xs = [x, x]
        ys = [y, y + 1]
        plot_line([x, x], [y, y + 1])

    plt.savefig('board_%02i.png' % name, bbox_inches='tight', dpi=100)
    plt.close()


def create_board(quadrants):
    """ 
    Prepare a board from a random select of each color's quadrant
    """
    board = {'tiles': {}, 'hwalls': [], 'vwalls': []}

    # Prepare a random order of colors
    colors = ['red', 'blue', 'green', 'yellow']
    shuffle(colors)

    # Prepare board
    for color_i, color in enumerate(colors):

        # Figure out the angle we're rotating by and the necessary offsets
        # This is a lazy approach to rotation
        rot = color_i * 90 / 180 * np.pi
        flip_walls = color_i in [1, 3]
        offset = np.array([color_i in [2, 3], color_i in [1, 2]], dtype=np.int) * -1
        rot_matrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])

        # Pick a quadrant
        quadrant_index = np.random.randint(len(quadrants[color]))
        quadrant = quadrants[color][quadrant_index]

        # Get new positions for each tile
        for tile in quadrant['tiles']:
            board['tiles'][tile] = np.matmul(quadrant['tiles'][tile], rot_matrix) + offset

        # Get new positions for each wall
        wall_names = ['hwalls', 'vwalls']
        for wall_i, wall in enumerate(wall_names):
            for x, y in quadrant[wall]:
                new_wall_i = (wall_i + flip_walls) % 2
                target = wall_names[new_wall_i]
                new_xy = np.matmul([x, y], rot_matrix)
                new_xy[new_wall_i] += offset[new_wall_i]
                board[target].append(new_xy)

    return board

if __name__ == "__main__":

    quadrants = load_quadrants()
    board = create_board(quadrants)
    plot_board(board)
