from subprocess import check_output

def gpu_vs_cpu():
    top = 30
    gpu_data = []
    cpu_data = []
    gtxt = open("gpu_time.txt", "w")
    ctxt = open("cpu_time.txt", "w")
    for x in xrange(32):
        avg_g = 0
        avg_c = 0
        print x
        for _ in xrange(top):
            g, c = map(lambda y: float(y), check_output(["./matrix_sum", "%d"%(512*(x+1)), "%d"%(32*(x+1)), "32"]).split())
            avg_g += g
            avg_c += c
        gtxt.write("%f\n"%(avg_g/float(top)))
        ctxt.write("%f\n"%(avg_c/float(top)))

    gtxt.close()
    ctxt.close()

def gpu_shift():
    for x in xrange(1024):
        check_output(["./matrix_sum", "1024", "%d"%(1024+x), "%d"%(1024-x)])


if __name__ == '__main__':

