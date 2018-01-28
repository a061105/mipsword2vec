#ifndef LBFGS_SOLVE_H
#define LBFGS_SOLVE_H

#include "util.h"
#include "multi.h"
#include "function.h"
#include "solver.h"

/////////LBFGS library/////////////
#include <Eigen/Core>
#include <LBFGS.h>
using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace LBFGSpp;
///////////////////////////////////

using namespace std;

class LBFGSSolve: public Solver{
		public:
		LBFGSSolve(double _tol, int _max_iter, int _m);
		void minimize(Function* func, double* U, double* V);
		
		class FuncWrap{
				public:
				FuncWrap(Function* _func, double* _U, double* _V){
						func = _func;
						U = _U;
						V = _V;
						iter = 0;
						prev_fval = 1e300;
						train_start_time = omp_get_wtime();
				}
				double operator()(const VectorXd& x, VectorXd& g){
						int d1 = func->dim_U();
						int d2 = func->dim_V();	
						
						//function value
						for(int i=0;i<d1;i++)
								U[i] = x[i];
						for(int i=0;i<d2;i++)
								V[i] = x[d1+i];
						double fval = func->fun();
						
						if( fval < prev_fval ){
								cerr << "iter=" << iter << ", fun_val=" << fval << ", train_time=" << omp_get_wtime()-train_start_time << ", prob_size=" << avg_prob_size << endl;
								iter++;
								prev_fval = fval;
						}else{
								cerr << "line search: fun_val=" << fval << ", train_time=" << omp_get_wtime()-train_start_time  << ", prob_size=" << avg_prob_size << endl;
						}
						
						//gradient
						double* gt = new double[d1+d2];
						func->grad_U( gt + 0 );
						func->grad_V( gt + d1 );
						for(int i=0;i<d1+d2;i++)
								g[i] = gt[i];
						delete[] gt;
						
						return fval;
				}

				private:
				Function* func;
				double* U;
				double* V;
				int iter;
				double prev_fval;
				double train_start_time;
		};

		private:
		int m;
		int max_iter;
		double tol;
		Function* func;
};

LBFGSSolve::LBFGSSolve(double _tol, int _max_iter, int _m){
		
		tol = _tol;
		max_iter = _max_iter;
		m = _m;
}


void LBFGSSolve::minimize(Function* func, double* U, double* V){
		
		int d1 = func->dim_U();
		int d2 = func->dim_V();
		
		//initialization
		VectorXd x = VectorXd::Zero(d1+d2);
		for(int i=0;i<d1;i++)
				x[i] = U[i];
		for(int i=0;i<d2;i++)
				x[d1+i] = V[i];
		
		//Function Wrapper
		FuncWrap  funcWrap(func, U, V);
		
		//solve
		LBFGSParam<double> param;
		param.epsilon = tol;
		param.max_iterations = max_iter;
		param.m = m;

		LBFGSSolver<double> solver(param);
		
		double fval;
		solver.minimize( funcWrap, x, fval);
}

#endif
