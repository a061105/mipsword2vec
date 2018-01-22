#ifndef GD_H
#define GD_H

#include "util.h"
#include "multi.h"
#include "function.h"
#include "solver.h"

using namespace std;

class GDSolve: public Solver{
		public:
		GDSolve(double _init_step_size, int _num_epoch);
		void minimize(Function* func, double* U, double* V);
		
		private:
		int num_epoch;
		double init_step_size;
};


GDSolve::GDSolve(double _init_step_size, int _num_epoch){
		
		num_epoch = _num_epoch;
		init_step_size = _init_step_size;
}

double train_time = 0.0;

void GDSolve::minimize(Function* func, double* U, double* V){

		int d1 = func->dim_U();
		int d2 = func->dim_V();
		
		double* g_U = new double[d1];
		double* g_V = new double[d2];

		int iter = 0;
		double fval = func->fun();
		int l=0;
		while( iter < num_epoch ){

				cerr << "iter=" << iter << ", fun_val=" << fval << ", train-time=" << train_time << ", #linesearch=" << l << endl;
				
				train_time -= omp_get_wtime();
				
				//compute gradient
				func->grad_U( g_U );
				func->grad_V( g_V );

				//update
				double eta = (double)init_step_size;
				for(l=0;l<10;l++){
						
						for(int i=0;i<d1;i++)
								U[i] -= eta * g_U[i];
						for(int i=0;i<d2;i++)
								V[i] -= eta * g_V[i];
						
						double fval_new = func->fun();
						if( fval_new < fval ){
								fval = fval_new;
								break;
						}
						
						for(int i=0;i<d1;i++)
								U[i] += eta * g_U[i];
						for(int i=0;i<d2;i++)
								V[i] += eta * g_V[i];
						
						eta /= 10;
				}
				if( l==10 ){
						cerr << "linesearch too many times" << endl;
						exit(0);
				}

				
				iter++;
				train_time += omp_get_wtime();
		}
		
		delete[] g_U;
		delete[] g_V;
}

#endif
