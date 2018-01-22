#ifndef FUNCTION_H
#define FUNCTION_H

class Function{

		public:
		virtual int dim_U()=0;
		virtual int dim_V()=0;
		virtual void grad_U(double* U)=0;
		virtual void grad_V(double* V)=0;
		virtual double fun()=0; //forward
};

#endif
