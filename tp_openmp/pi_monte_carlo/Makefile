EXTRAFLAGS:=  -pedantic -std=c99 -ggdb
SERIAL=pi_monte_carlo_serial
MP=pi_monte_carlo_mp


.PHONY: all clean

$(SERIAL): $(SERIAL).c 
	gcc  -o $@ $(EXTRAFLAGS) $< 

$(MP): $(MP).c 
	gcc  -fopenmp -o $@ $(EXTRAFLAGS) $< 

%.o: %.c
	gcc -c $(EXTRAFLAGS) -o $@ $<

all: $(SERIAL) $(MP) 

clean:
	rm -rf $(SERIAL) $(MP) *.o *.dSYM

