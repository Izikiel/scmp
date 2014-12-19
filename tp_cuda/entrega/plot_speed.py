import sys
import matplotlib.pyplot as plt

def parse_time(filename):
    with open(filename, "r") as f:
        data = []
        pos = []
        for i, l in enumerate(f):
            data.append(float(l.strip()))
            pos.append(((i+1)*512)**2)
        return data, pos

def plot_vs(files):
    fig = plt.figure()
    for f in files:
        data, pos = parse_time(f)
        ax = fig.add_subplot(111)
        ax.plot(pos, data)
        ax.yaxis.set_label_text("Tiempo (mS)")
        ax.xaxis.set_label_text("Tamanio matriz")
    plt.autoscale(enable=True, axis=u'both', tight=True)
    plt.legend(("Mono Core", "GPU"), loc=2)
    plt.savefig("cpu_gpu.png")

def plot_speedup(files):
    assert len(files) > 1
    mono, dual = files

    mono, pos = parse_time(mono)
    dual, _ = parse_time(dual)

    speedup = map(lambda (x,y): float(y)/x, zip(dual, mono))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pos, speedup)
    ax.yaxis.set_label_text("Speedup")
    ax.xaxis.set_label_text("Tamanio matriz")
    ax.set_xscale("log")
    plt.autoscale(enable=True, axis=u'both', tight=True)
    plt.legend(("GPU / Mono Core", ""), loc=0)
    plt.savefig("speedup.png")



if __name__ == '__main__':
    plot_vs(sys.argv[1:])
    if len(sys.argv) > 2:
        plot_speedup(sys.argv[1:])