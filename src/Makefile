CC = cc
CFLAGS = -O3 -Wall -W -ffast-math
LDFLAGS = -lm

all: train play benchmark

train: train.c common.h
	$(CC) train.c -o train $(CFLAGS) $(LDFLAGS)

play: play.c common.h
	$(CC) play.c -o play $(CFLAGS) $(LDFLAGS)

benchmark: benchmark.c common.h
	$(CC) benchmark.c -o benchmark $(CFLAGS) $(LDFLAGS)

clean:
	rm -f train play benchmark
