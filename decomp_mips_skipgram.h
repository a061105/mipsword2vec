#ifndef DECOMP_MIPS_SKIPGRAM_H
#define DECOMP_MIPS_SKIPGRAM_H

#include "util.h"
#include "multi.h"
#include "function.h"
#include "mips.h"

class DecompMipsSkipgram: public Function{
		
		public:
		DecompMipsSkipgram(Problem* _prob, Model* _model, int _query_size, int _factor_dim, int _num_cluster, double _prob_discard, int _nthread){
				prob = _prob;
				model = _model;
				R = model->R;
				K = model->K;
				nthread = _nthread;

				//N by K Sparse Matrix for loss gradients
				pred_prob.resize(model->K);
				
				//parameters related to MIPS
				query_size = _query_size;
				factor_dim = _factor_dim;
				num_cluster = _num_cluster;
				prob_discard = _prob_discard;
				
				//construct data structures
				num_factor = (int) ceil( (double)model->R / factor_dim );
				
				cerr << "prob_discard=" << prob_discard << ", num_factor=" << num_factor << ", |query|=" << query_size << ", num_cluster=" << num_cluster << endl;

				mips = NULL;
				
				g_sum.resize(nthread);
				for(int i=0;i<nthread;i++)
						g_sum[i] = new ArrayHash(model->K);
		}
		
		~DecompMipsSkipgram(){
				
				if( mips != NULL )
						delete mips;
				
				for(int i=0;i<nthread;i++)
						delete g_sum[i];
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
				
				reconstruct_mips(V, K, R);

				double fun_val = 0.0;
				#pragma omp parallel for
				for(int n=0;n<prob->data.size();n++){
						int i = prob->data[n]->first;
						double fea_fq = prob->data[n]->second;
						SparseVec* lab_fq = prob->labels[n];
						double* ui = U+i*R;
						//MIPS query
						vector<vector<int> > q_result;
						int num_return = max(query_size, (int)lab_fq->size());
						mips->query( ui, q_result, num_return );
						
						if( CC>=2 ){
								vector<int> q2 = q_result[0];
								distinct_vec(q2);
								cerr << q2.size() << endl;
								exit(0);
						}
						//compute sum_j qij * <ui,vj>
						double emp_sum = 0.0;
						for(SparseVec::iterator it=lab_fq->begin(); it!=lab_fq->end(); it++)
								emp_sum += it->second * inner_prod(ui, V+it->first*R, R);
						//compute probabilities 
						double logSum = compute_partition_fun( q_result, ui, V, pred_prob[i] );
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
		
		double compute_partition_fun( vector<vector<int>>& q_result, double* ui, double* V, SparseVec& prob_i ){
				
				//compute factorwise probabilities
				vector<SparseVec> fac_prob(num_factor);
				for(int f=0;f<num_factor;f++){
						
						int f_offset = f*factor_dim;
						double* uif = ui+f_offset;
						for(vector<int>::iterator it=q_result[f].begin(); it!=q_result[f].end(); it++){
								int j = *it;
								double val = inner_prod(uif,V+j*R+f_offset,factor_dim)*num_factor;
								fac_prob[f].push_back( make_pair(j, val ) );
						}
						
						double logZ_f = make_multinomial( fac_prob[f] );
						//filter those with small probability
						sort( fac_prob[f].begin(), fac_prob[f].end(), ValueComp() );
						double cumul=0.0;
						for(int r=fac_prob[f].size()-1;r>=0;r--){
								cumul += fac_prob[f][r].second;
								if( cumul > prob_discard )
										break;
								fac_prob[f].pop_back();
						}
				}
				
				//merge
				int t = omp_get_thread_num();
				g_sum[t]->clear();
				for(int f=0;f<num_factor;f++)
						for(auto it=fac_prob[f].begin(); it!=fac_prob[f].end(); it++){
								g_sum[t]->add( it->first, it->second );
						}
				g_sum[t]->to_sparse_vec( prob_i );
				
				for(auto it=prob_i.begin(); it!=prob_i.end(); it++){
						int j = it->first;
						it->second = inner_prod( ui, V+j*R, R );
				}
				double logZ = make_multinomial( prob_i );

				return logZ;
		}

		/** (re)build MIPS data structure for V: K*R
		 */
		void reconstruct_mips(double* V, int K, int R){
				
				if( mips == NULL )
						mips = new RandomClusterMIPS( factor_dim, num_factor, num_cluster );
				else
						mips->clear();
				
				#pragma omp parallel for
				for(int f=0;f<num_factor;f++){
						for(int k=0;k<K;k++){
								mips->insert( k, f, V+k*R );
						}
				}
		}
		
		
		Problem* prob;
		Model* model;
		int R;
		int K;
		int nthread;
		
		//updated when fun() is called
		SparseMat pred_prob;//N by K
		
		//hyperparameters related to MIPS
		int query_size;
		int factor_dim;
		int num_cluster;
		double prob_discard;//probability mass in loss grad that could be ignored

		//derived
		int num_factor;
		FactorizedMIPS* mips;
		vector<ArrayHash*> g_sum;
};

#endif
