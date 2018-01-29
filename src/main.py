#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import yaml

def load_grids(path="grids.yaml"):
    """ 
    Loads the grids yaml and automatically populates center/edge walls
    Returns the grids as a dictionary by color
    """
    with open("grids.yaml") as f:
        grids = yaml.load(f)

    for color in grids:
        for grid in grids[color]:
            grid['hwalls'].append([0, 1])
            grid['vwalls'].append([1, 0])
            for i in range(8):
                grid['hwalls'].append([i, 8])
                grid['vwalls'].append([8, i])
    return grids


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


def plot_grid(grid, name=''):
    """
    Plots a given grid for verification purposes
    """
    # Prepare
    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    plt.axis([0, 8, 8, 0])
    plt.grid()

    for tile in grid['tiles']:
        x, y = grid['tiles'][tile]
        plot_marker(x, y, tile)

    for x, y in grid['hwalls']:
        plot_line([x, x + 1], [y, y])

    for x, y in grid['vwalls']:
        xs = [x, x]
        ys = [y, y + 1]
        plot_line([x, x], [y, y + 1])

    plt.savefig('grid_%s.png' % name, bbox_inches='tight', dpi=100)
    plt.close()


if __name__ == "__main__":
    grids = load_grids()
    for color in grids:
        for i, grid in enumerate(grids[color]):
            name = '%s%i' % (color[0], i)
            plot_grid(grid, name)
