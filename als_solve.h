#ifndef ALS_SOLVE_H
#define ALS_SOLVE_H

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

class AlsSolve: public Solver{
		public:
				AlsSolve(double _tol, int _max_iter, int _max_inner_iter, int _m);
				void minimize(Function* func, double* U, double* V);

		private:

				class FuncWrap_U{
						public:
								FuncWrap_U(Function* _func, double* _U, double _train_start_time){
										func = _func;
										U = _U;
										iter = 0;
										prev_fval = 1e300;
										train_start_time = _train_start_time;
								}
								double operator()(const VectorXd& x, VectorXd& g){
										int d = func->dim_U();
										//function value
										for(int i=0;i<d;i++)
												U[i] = x[i];
										double fval = func->fun();

										if( fval < prev_fval ){
												cerr << "iter=" << iter << ", fun_val=" << fval << ", train_time=" << omp_get_wtime()-train_start_time << ", prob_size=" << avg_prob_size << endl;
												iter++;
												prev_fval = fval;
										}else{
												cerr << "line search: fun_val=" << fval << ", train_time=" << omp_get_wtime()-train_start_time  << ", prob_size=" << avg_prob_size << endl;
										}

										//gradient
										double* gt = new double[d];
										func->grad_U( gt );
										for(int i=0;i<d;i++)
												g[i] = gt[i];
										delete[] gt;

										return fval;
								}

						private:
								Function* func;
								double* U;
								int iter;
								double prev_fval;
								double train_start_time;
				};

				class FuncWrap_V{
						public:
								FuncWrap_V(Function* _func, double* _V, double _train_start_time){
										func = _func;
										V = _V;
										iter = 0;
										prev_fval = 1e300;
										train_start_time = _train_start_time;
								}
								double operator()(const VectorXd& x, VectorXd& g){
										int d = func->dim_V();
										//function value
										for(int i=0;i<d;i++)
												V[i] = x[i];
										double fval = func->fun();

										if( fval < prev_fval ){
												cerr << "iter=" << iter << ", fun_val=" << fval << ", train_time=" << omp_get_wtime()-train_start_time << ", prob_size=" << avg_prob_size << endl;
												iter++;
												prev_fval = fval;
										}else{
												cerr << "line search: fun_val=" << fval << ", train_time=" << omp_get_wtime()-train_start_time  << ", prob_size=" << avg_prob_size << endl;
										}

										//gradient
										double* gt = new double[d];
										func->grad_V( gt );
										for(int i=0;i<d;i++)
												g[i] = gt[i];
										delete[] gt;

										return fval;
								}
						private:
								Function* func;
								double* V;
								int iter;
								double prev_fval;
								double train_start_time;
				};

		private:
				int m;
				int max_iter;
				int max_inner_iter;
				double tol;
				Function* func;
				double train_start_time;
};


AlsSolve::AlsSolve(double _tol, int _max_iter, int _max_inner_iter, int _m){

		tol = _tol;
		max_iter = _max_iter;
		max_inner_iter = _max_inner_iter;
		m = _m;
		train_start_time = omp_get_wtime();
}


void AlsSolve::minimize(Function* func, double* U, double* V){

		int d1 = func->dim_U();
		int d2 = func->dim_V();
		
		//initialization
		VectorXd x_U = VectorXd::Zero(d1);
		VectorXd x_V = VectorXd::Zero(d2);
		for(int i=0;i<d1;i++)
				x_U[i] = U[i];
		for(int i=0;i<d2;i++)
				x_V[i] = V[i];

		//Function Wrapper
		FuncWrap_U f_U(func, U, train_start_time);
		FuncWrap_V f_V(func, V, train_start_time);
		
		//solve
		LBFGSParam<double> param;
		param.epsilon = tol;
		param.m = m;
		param.max_iterations = max_inner_iter;
		
		LBFGSSolver<double> solver(param);
		
		double fval;
		for(int iter=0;iter<max_iter;iter++){

				cerr << "ALS-iter=" << iter << ", solving U..." << endl;
				solver.minimize( f_U, x_U, fval);
				cerr << "ALS-iter=" << iter << ", solving V..." << endl;
				solver.minimize( f_V, x_V, fval);
		}
}

#endif
