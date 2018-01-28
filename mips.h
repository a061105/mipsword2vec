#ifndef MIPS_H
#define MIPS_H

#include "util.h"

const int MAX_LABEL_SIZE = 2000000;
//const int NUM_SORT_CLUSTER = 100;

class FactorizedMIPS{
		
			public:
			//get results of each factor
			virtual void query( double* q, vector<vector<int> >& result, int size )=0;
			virtual void insert( int id, int f, double* x )=0;
			virtual void remove( int id, int f )=0;
			virtual void clear()=0;
};

class RandomClusterMIPS: public FactorizedMIPS{
		
			public:
			RandomClusterMIPS(int _factor_dim, int _num_factor, int _num_cluster){
					
					factor_dim = _factor_dim; //factor dim
					num_factor = _num_factor;
					num_cluster = _num_cluster; //number of cluster
					
					int D = factor_dim * num_factor;
					centers.resize(num_cluster);
					for(int k=0;k<num_cluster;k++){
							centers[k] = new double[D];
							for(int j=0;j<D;j++){
									centers[k][j] = (2.0*randu()) - 1.0;
									//centers[k][j] = randn();
							}
							for(int f=0;f<num_factor;f++)
									normalize(centers[k]+f*factor_dim, factor_dim);
					}
					
					cluster_members.resize(num_cluster);
					for(int k=0;k<num_cluster;k++){
							cluster_members[k].resize(num_factor);
					}

					cluster_id_map = new pair<int,int>*[MAX_LABEL_SIZE];
					for(int i=0;i<MAX_LABEL_SIZE;i++){
							cluster_id_map[i] = new pair<int,int>[num_factor];
							for(int f=0;f<num_factor;f++){
									cluster_id_map[i][f].first = -1;
									cluster_id_map[i][f].second = -1;
							}
					}
			}
			
			~RandomClusterMIPS(){
					
					for(int i=0;i<MAX_LABEL_SIZE;i++)
							delete[] cluster_id_map[i];
					delete[] cluster_id_map;
			}
					
			void clear(){

					for(int i=0;i<MAX_LABEL_SIZE;i++)
							for(int f=0;f<num_factor;f++){
									cluster_id_map[i][f].first = -1;
									cluster_id_map[i][f].second = -1;
							}
					
					for(int k=0;k<num_cluster;k++){
							for(int f=0;f<num_factor;f++)
									cluster_members[k][f].clear();
					}
			}

			void insert( int id, int f, double* x ){
					
					int f_offset = f*factor_dim;
					
					int argmax;
					double max_prod = -1e300;
					for(int k=0;k<centers.size();k++){
							double prod = inner_prod(centers[k]+f_offset, x+f_offset, factor_dim);
							if( prod > max_prod ){
									max_prod = prod;
									argmax = k;
							}
					}
					int cid = argmax;
					int pos = (int)cluster_members[cid][f].size();
					cluster_members[cid][f].push_back(id);
					
					cluster_id_map[id][f].first = cid;
					cluster_id_map[id][f].second = pos;
			}

			void remove( int id, int f ){
				
					int cid = cluster_id_map[id][f].first;
					int pos = cluster_id_map[id][f].second;
					cluster_id_map[id][f].first = -1;
					cluster_id_map[id][f].second = -1;

					int last_ind = (int)cluster_members[cid][f].size()-1;
					int swap_member = cluster_members[cid][f][last_ind];
					swap( cluster_members[cid][f][pos], cluster_members[cid][f][last_ind]);
					cluster_id_map[swap_member][f].second = pos;

					cluster_members[cid][f].pop_back();
			}
			
			
			void query( double* q, vector<vector<int> >& result, int size ){
					
					//compute scores of each cluster
					vector<int>* index = new vector<int>[num_factor];
					for(int f=0;f<num_factor;f++){
							index[f].resize(num_cluster);
							for(int k=0;k<num_cluster;k++)
									index[f][k] = k;
					}
					
					double** score = new double*[num_cluster];
					for(int k=0;k<num_cluster;k++){
							score[k] = new double[num_factor];
							for(int f=0;f<num_factor;f++)
									score[k][f] = 0.0;
							for(int j=0;j<num_factor*factor_dim;j++){
									int f = j / factor_dim;
									score[k][f] += centers[k][j] * q[j];
							}
					}
					
					double** score_tr = new double*[num_factor];
					for(int f=0;f<num_factor;f++)
							score_tr[f] = new double[num_cluster];
					transpose( score, num_cluster, num_factor, score_tr );
					
					//sort and get query result within top cluster
					result.resize(num_factor);
					for(int f=0;f<num_factor;f++){
							
							ScoreComp comp(score_tr[f]);
							/*if( num_cluster > NUM_SORT_CLUSTER ){
									nth_element(index[f].begin(), index[f].begin()+NUM_SORT_CLUSTER, index[f].end(), comp);
									sort(index[f].begin(), index[f].begin()+NUM_SORT_CLUSTER, comp);
							}else{*/
									sort(index[f].begin(), index[f].end(), comp);
							//}
							
							result[f].clear();
							int r =0;
							while( r < index[f].size() && result[f].size() < size ){
									result[f].insert( result[f].end(), cluster_members[ index[f][r] ][f].begin(), cluster_members[ index[f][r] ][f].end() );
									r++;
							}
					}
					
					for(int k=0;k<num_cluster;k++)
							delete[] score[k];
					delete[] score;
					for(int f=0;f<num_factor;f++)
							delete[] score_tr[f];
					delete[] score_tr;
					delete[] index;
			}


			private:
			int num_cluster;
			vector<double*> centers;
			
			int factor_dim;
			int num_factor;
			vector<vector< vector<int> > >  cluster_members; //num_cluster * num_factor * cluster_size
			pair<int,int>** cluster_id_map; // an array map from (instance id, factor id) to (cluster id, pos_in_cluster_members)
};


#endif
