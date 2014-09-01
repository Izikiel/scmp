EXTRAFLAGS:= -Wall -Werror -pedantic --std=c99 -ggdb -O0
EXE=explicit

.PHONY: all clean

$(EXE): $(EXE).c 
	gcc  -o $@ $(EXTRAFLAGS) $< 


%.o: %.c
	gcc -c $(EXTRAFLAGS) -o $@ $<

all: $(EXE) 

clean:
	rm -f $(EXE) *.o *.dSYM

