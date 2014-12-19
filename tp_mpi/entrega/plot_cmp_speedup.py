import sys
import matplotlib.pyplot as plt

def parse_150(filename):
    with open(filename, "r") as f:
        data = []
        pos = []
        i = 20
        for _ in xrange(150):
            data.append(int(f.readline().strip()))
            pos.append(i)
            i += 10
        return data, pos

def plot_grid(files):
    fig = plt.figure()
    for f in files:
        data, pos = parse_150(f)
        ax = fig.add_subplot(111)
        ax.plot(pos, data)
        # ax.legend("pepe")
        ax.axis([20, 1500, 20, 40000])
        ax.yaxis.set_label_text("Tiempo (uS)")
        ax.xaxis.set_label_text("Tamanio grilla")
    plt.legend(("Mono Core", "Dual Core"), loc=2)
    plt.savefig("dual_mono.png")
    # plt.show()

def plot_speedup(files):
    assert len(files) > 1
    mono, dual = files

    mono, pos = parse_150(mono)
    dual, _ = parse_150(dual)

    speedup = map(lambda (x,y): float(y)/x, zip(dual, mono))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pos, speedup)
    ax.axis([20, 1500, 0, 2])
    ax.yaxis.set_label_text("Speedup")
    ax.xaxis.set_label_text("Tamanio grilla")
    plt.legend(("Dual Core / Mono Core", ""), loc=2)
    # plt.show()
    plt.savefig("speedup.png")
        



if __name__ == '__main__':
    plot_grid(sys.argv[1:])
    if len(sys.argv) > 2:
        plot_speedup(sys.argv[1:])