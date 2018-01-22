CXX = g++
CFLAGS = -O3 -std=c++11 -fopenmp
LIB = include/

all: clean train

train: train.cpp
	$(CXX) $(CFLAGS) -I$(LIB) -o train train.cpp

clean:
	rm -f train
