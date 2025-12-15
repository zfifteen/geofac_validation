CC=gcc
GMP_INCLUDE=-I/opt/homebrew/include
MPFR_INCLUDE=-I/opt/homebrew/include
GMP_LIB=-L/opt/homebrew/lib -lgmp
MPFR_LIB=-L/opt/homebrew/lib -lmpfr

CFLAGS=-O2 -Wall -I../include $(GMP_INCLUDE) $(MPFR_INCLUDE) -DMPFR_PREC=4000
LDFLAGS=-L../build -lz5d_predictor $(GMP_LIB) $(MPFR_LIB)

all: z5d_adapter

z5d_adapter: z5d_adapter.c
	$(CC) $(CFLAGS) -o ../z5d_adapter z5d_adapter.c $(LDFLAGS)

clean:
	rm -f ../z5d_adapter
