from subprocess import check_output
import sys

def cpu_run(exe, out_txt):
    top = 100
    gpu_data = []
    cpu_data = []
    with open(out_txt, "w", 0) as txt:
        for x in xrange(50):
            avg = 0
            print x
            i = 0
            while i < top:
                val = int(check_output([exe, "%d"%(x+1), ]).strip())
                if val > 0:
                    avg += val
                    i += 1
            txt.write("%d\n"%(avg/top))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "./exe out.txt"
    else:
        cpu_run(sys.argv[1], sys.argv[2])
