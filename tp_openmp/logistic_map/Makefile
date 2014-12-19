EXTRAFLAGS:= --std=c99 -ggdb
SERIAL=logistic_map_serial
MP=logistic_map_mp


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

