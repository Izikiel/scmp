.PHONY: all clean
MAT_SUM=matrix_sum
UTILS=utils.o
HEADERS= utils.h

%.o: %.cu
	nvcc -g -G -lineinfo -c -o $@ $<

$(MAT_SUM): $(MAT_SUM).cu $(UTILS) $(HEADERS)
	nvcc -o $@ $< $(UTILS)

all: $(MAT_SUM)

clean:
	rm -f $(MAT_SUM)
	rm -f *.o