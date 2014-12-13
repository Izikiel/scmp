import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab

def parse_data(filename):
    with open(filename, "r") as f:
        x = []
        y = []
        z = []
        i = 1
        for l in f:
            values = map(float, l.strip().split(" "))
            for pos, val in enumerate(values):
                x.append(pos)
                y.append(i)
                z.append(val)
            i += 1
    return (x,y,z)

def plot(filename, show=0):
    x,y,z = parse_data(filename)
    fig = plt.figure(figsize=(10,10), linewidth=1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x,y,z, "x")
    t = "Concentration"
    p = "Position"
    time = "Time"
    ax.set_xlabel(p)
    ax.set_ylabel(time)
    ax.set_zlabel(t)
    if show:
        plt.show()
    else:
        plt.savefig("out.svg")


if __name__ == '__main__':
    if len(sys.argv) == 3:
        plot(sys.argv[1], True)
    elif len(sys.argv) == 2:
        plot(sys.argv[1], False)
