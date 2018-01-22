CXX = g++
CFLAGS = -O3 -std=c++11 -fopenmp
LIB = include/

all: clean train evaluate

train: train.cpp
	$(CXX) $(CFLAGS) -I$(LIB) -o train train.cpp

evaluate: evaluate.cpp
	$(CXX) $(CFLAGS) -o evaluate evaluate.cpp

clean:
	rm -f train evaluate
