import sys

import matplotlib.pyplot as plt
from math import log

l10 = lambda x: int(log(x, 10)) if x > 9 else 0
zeroes = "0000000"

def merge_files(file_a, file_b):
	with open(file_a, "r") as a:
		with open(file_b, "r") as b:
			with open("merge_out.txt", "w") as out:
				for x, y in zip(a,b):
					out.write(x.strip() + " " + y)


def parse_data(filename):
	data = []
	with open(filename, "r") as f:
		for l in f:
			data.append(map(float, l.strip().split()))
	return data


def plot_steps(filename):
	data = parse_data(filename)
	i = 0
	fig = plt.figure()
	for l in data:
		ax = fig.add_subplot(111)
		ax.plot(l)
		ax.axis([0, 20, -1, 2])
		ax.yaxis.set_label_text("Concentration")
		ax.xaxis.set_label_text("Position")
		fig.savefig("step_%s%i.png"%(zeroes[l10(i):],i))
		fig.clf()
		i += 1


if __name__ == '__main__':
	if len(sys.argv) == 3:
		merge_files(sys.argv[1], sys.argv[2])
		plot_steps("merge_out.txt")
	elif len(sys.argv) == 2:
		plot_steps(sys.argv[1])

