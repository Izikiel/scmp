from subprocess import check_output
import sys
def cpu_run(exe, out_txt):
    top = 100
    gpu_data = []
    cpu_data = []
    with open(out_txt, "w", 0) as txt:
        x = 10**9
        while x < 10**11:
            avg = 0
            print x
            for _ in xrange(top):
                avg += int(check_output([exe, "%d"%(x), ]).strip())
            txt.write("%d\n"%(avg/top))
            x *= 10

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "./exe out.txt"
    else:
        cpu_run(sys.argv[1], sys.argv[2])
