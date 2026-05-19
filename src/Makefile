CC = cc
CFLAGS = -O3 -Wall -W -ffast-math
LDFLAGS = -lm

all: train play

train: train.c common.h
	$(CC) train.c -o train $(CFLAGS) $(LDFLAGS)

play: play.c common.h
	$(CC) play.c -o play $(CFLAGS) $(LDFLAGS)

clean:
	rm -f train play gomoku_nn.bin
