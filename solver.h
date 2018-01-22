#ifndef SOLVER_H
#define SOLVER_H

#include "function.h"

class Solver{
	
		public:
		virtual void minimize(Function* func, double* U, double* V)=0;
};

#endif
