.PHONY: all clean
R_ARR=reverse_array
R_MAT=reverse_matrix
MAT_SUM=matrix_sum
UTILS=utils.o
HEADERS= utils.h

%.o: %.cu
	nvcc -c -o $@ $<

$(R_ARR): $(R_ARR).cu $(UTILS) $(HEADERS)
	nvcc -o $@ $< $(UTILS)

$(R_MAT): $(R_MAT).cu $(UTILS) $(HEADERS)
	nvcc -o $@ $< $(UTILS)

$(MAT_SUM): $(MAT_SUM).cu $(UTILS) $(HEADERS)
	nvcc -o $@ $< $(UTILS)

all: $(R_ARR) $(R_MAT) $(MAT_SUM)

clean:
	rm -f $(R_ARR)
	rm -f $(R_MAT)
	rm -f $(MAT_SUM)
	rm -f *.o