LFLAGS=-lm
CC=mpicc
CFLAGS=-O3

all: 
	$(CC) $(CFLAGS) -o jacobi jacobi.c $(LFLAGS)

clean:
	rm -f jacobi *.o
