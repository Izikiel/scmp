import sys

import matplotlib.pyplot as plt
from math import log
from joblib import Parallel, delayed
from multiprocessing import cpu_count

l10 = lambda x: int(log(x, 10)) if x > 9 else 0
zeroes = "000"

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

def plot_data_line(l, i):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(l)
	ax.axis([0, 20, -1, 2])
	ax.yaxis.set_label_text("Concentracion (g/ml)")
	ax.xaxis.set_label_text("Posicion (cm)")
	if i < 1000:
		fig.savefig("step_%s%i.png"%(zeroes[l10(i):],i))
	else:
		fig.savefig("step_%i.png"%(i))
	plt.close(fig)
	# fig.clf()

def plot_steps(filename):
	data = parse_data(filename)
	# Parallel(n_jobs=cpu_count())(delayed(plot_data_line)(l, i) for i, l in enumerate(data))

	for i, l in enumerate(data):
		plot_data_line(l, i)


if __name__ == '__main__':
	if len(sys.argv) == 3:
		merge_files(sys.argv[1], sys.argv[2])
		plot_steps("merge_out.txt")
	elif len(sys.argv) == 2:
		plot_steps(sys.argv[1])

