EXTRAFLAGS:= -Wall -Werror -pedantic --std=c99 -ggdb -O0
EXE=difusion_solvers

.PHONY: all clean

$(EXE): $(EXE).c 
	gcc  -o $@ $(EXTRAFLAGS) $< 


%.o: %.c
	gcc -c $(EXTRAFLAGS) -o $@ $<

all: $(EXE) 

clean:
	rm -rf $(EXE) *.o *.dSYM

