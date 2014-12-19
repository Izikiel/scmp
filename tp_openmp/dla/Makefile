EXTRAFLAGS:= -pedantic -lm -std=gnu99 -ggdb
SERIAL=dla_serial
THREADS=dla_threads


.PHONY: all clean

$(SERIAL): $(SERIAL).o 
	gcc $< -o $@ $(EXTRAFLAGS)

$(THREADS): $(THREADS).o 
	gcc $< -lpthread -o $@ $(EXTRAFLAGS)

%.o: %.c
	gcc $< -c $(EXTRAFLAGS) -o $@ 

all: $(SERIAL) $(THREADS)

clean:
	rm -rf $(SERIAL) $(THREADS) *.o *.dSYM

