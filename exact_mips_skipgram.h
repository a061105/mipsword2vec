#ifndef EXACT_MIPS_SKIPGRAM_H
#define EXACT_MIPS_SKIPGRAM_H

#include "util.h"
#include "multi.h"
#include "function.h"

class ExactMipsSkipgram: public Function{
		
		public:
		ExactMipsSkipgram(Problem* _prob, Model* _model, int _query_size){
					prob = _prob;
					model = _model;
					query_size = _query_size;
					pred_prob.resize(model->K);
		}
		
		~ExactMipsSkipgram(){
		}

		int dim_U() override {
				return model->K * model->R;
		}

		int dim_V() override {
				return model->K * model->R;
		}
		
		double fun() override {
				
				double* U = model->U; //N by R (N=K)
				double* V = model->V; //K by R
				int R = model->R;
				int K = model->K;
				
				double fun_val = 0.0;
				#pragma omp parallel for
				for(int n=0;n<prob->data.size();n++){
						int i = prob->data[n]->first;
						double fea_fq = prob->data[n]->second;
						SparseVec* lab_fq = prob->labels[n];
						
						//compute inner products u_i v_j for all j
						vector<double> pred(K);
						vector<int> index(K);
						for(int j=0;j<K;j++){
								
								pred[j] = inner_prod(U+i*R, V+j*R, R);
								index[j] = j;
						}
						sort(index.begin(), index.end(), ScoreComp(&(pred[0])) );
						
						//compute log_sum
						double sum = 0.0;
						double max_val = pred[index[0]];
						pred_prob[i].clear();
						for(int r=0;r< max(query_size,(int)lab_fq->size()) ;r++){
								int j = index[r];
								double v = exp(pred[j]-max_val);
								pred_prob[i].push_back( make_pair(j,v) );
								sum += v;
						}
						double logSum = log(sum) + max_val;
						
						//compute predicative probabilities
						for(SparseVec::iterator it=pred_prob[i].begin(); it!=pred_prob[i].end(); it++)
								it->second /= sum;
						
						//compute sum_j qij * <ui,vj>
						double emp_sum = 0.0;
						for(SparseVec::iterator it=lab_fq->begin(); it!=lab_fq->end(); it++)
								emp_sum += it->second * pred[it->first];
						
						double neg_loglike = fea_fq * (logSum - emp_sum);
						
						#pragma omp atomic update
						fun_val += neg_loglike;
				}

				return fun_val;
		}
		

		void grad_U(double* g_U) override {
				
				double* V = model->V;
				int K = model->K;
				int R = model->R;

				vector<pair<int,double>*>& data = prob->data;
				vector<SparseVec*>& labels = prob->labels;
				
				for(int i=0;i<K*R;i++)
						g_U[i] = 0.0;
				
				//gradient of partition function
				#pragma omp parallel for
				for(int n=0;n<data.size();n++){
						int i = data[n]->first;
						double fi = data[n]->second;
						for(int p=0;p<pred_prob[i].size();p++){
								int j = pred_prob[i][p].first;
								double weight_ij = fi * pred_prob[i][p].second;
								for(int r=0;r<R;r++)
										g_U[i*R+r] += weight_ij*V[j*R+r];
						}
				}

				//gradient of empirical distribution
				#pragma omp parallel for
				for(int n=0;n<prob->data.size();n++){
						int i = data[n]->first;
						double fi = data[n]->second;
						SparseVec& lab_fq = *(labels[n]);
						for(int p=0;p<lab_fq.size();p++){
								int j = lab_fq[p].first;
								double weight_ij = fi * lab_fq[p].second;
								for(int r=0;r<R;r++)
										g_U[i*R+r] -= weight_ij*V[j*R+r];
						}
				}
		}
		
		
		void grad_V(double* g_V) override {
				
				double* U = model->U; //K by R
				int K = model->K;
				int R = model->R;
				
				vector<pair<int,double>*>& data = prob->data;
				vector<SparseVec*>& labels = prob->labels;

				for(int i=0;i<K*R;i++)
						g_V[i] = 0.0;
				
				//gradient of partition function
				for(int n=0;n<data.size();n++){
						int i = data[n]->first;
						double fi = data[n]->second;
						#pragma omp parallel for
						for(int p=0;p<pred_prob[i].size();p++){
								int j = pred_prob[i][p].first;
								double weight_ij = fi * pred_prob[i][p].second;
								for(int r=0;r<R;r++)
										g_V[j*R+r] += weight_ij*U[i*R+r];
						}
				}

				//gradient of empirical distribution
				for(int n=0;n<prob->data.size();n++){
						int i = data[n]->first;
						double fi = data[n]->second;
						SparseVec& lab_fq = *(labels[n]);
						#pragma omp parallel for
						for(int p=0;p<lab_fq.size();p++){
								int j = lab_fq[p].first;
								double weight_ij = fi * lab_fq[p].second;
								for(int r=0;r<R;r++)
										g_V[j*R+r] -= weight_ij*U[i*R+r];
						}
				}
		}
		
		private:
		Problem* prob;
		Model* model;
		int query_size;

		//updated when fun() is called
		SparseMat pred_prob;//N by K
};

#endif
