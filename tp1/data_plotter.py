#!/usr/bin/python

import sys

def parse_data(filename):
	with open(filename, "r") as f:
		data = []
		i = 1
		for l in f:
			line = map(float, l.strip().split(" "))
			data.append((i, [(index, val) for index, val in enumerate(line)]))
			i += 1
		return data


if __name__ == '__main__':
	for x in parse_data(sys.argv[1]):
		print x