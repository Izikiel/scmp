import matplotlib.pyplot as plt
import sys


def parse_data(file):
	pos = []
	vals = []
	with open(file, "r") as f:
		for l in f:
			p, v = map(float, l.strip().split())
			pos.append(p)
			vals.append(v)
	return pos, vals


def plot_logistic_map(file):
	pos, vals = parse_data(file)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(pos, vals, ".")
	ax.yaxis.set_label_text("X")
	ax.xaxis.set_label_text("R")
	fig.savefig("logistic_map.png")

if __name__ == '__main__':
	if len(sys.argv) == 2:
		plot_logistic_map(sys.argv[1])