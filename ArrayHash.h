#ifndef ARRAYHASH_H
#define ARRAYHASH_H

#include <cassert>
#include <vector>
using namespace std;

const double NAN_VAL = -12121.0;

class ArrayHash{
		
		public:
		ArrayHash(int key_limit){

				capacity = key_limit;
				val_arr = new double[capacity];
				for(int i=0;i<capacity;i++)
						val_arr[i] = NAN_VAL;

		}

		~ArrayHash(){
				delete[] val_arr;
		}
		
		void add(int key, double val){
				if( val_arr[key] == NAN_VAL ){
						key_list.push_back(key);
						val_arr[key] = 0.0;
				}
				val_arr[key] += val;
		}
		
		void set(int key, double val){
				
				if( val_arr[key] == NAN_VAL ){
						key_list.push_back(key);
				}
				val_arr[key] = val;
		}
		
		bool exist(int key){
				return (val_arr[key] != NAN_VAL);
		}

		double get(int key){
				
				assert( val_arr[key] != NAN_VAL );
				return val_arr[key];
		}
		
		void clear(){
				
				for(vector<int>::iterator it=key_list.begin(); it!=key_list.end(); it++)
						val_arr[*it] = NAN_VAL;
				
				key_list.clear();
		}
		
		int size(){
				return key_list.size();
		}

		void to_sparse_vec( vector<pair<int,double> >& sv ){

				sv.clear();
				sv.reserve(key_list.size());
				for(vector<int>::iterator it=key_list.begin(); it!=key_list.end(); it++)
						sv.push_back( make_pair(*it , val_arr[*it]) );
		}
		
		void to_map( map<int,double>& m ){
				
				m.clear();
				for(vector<int>::iterator it=key_list.begin(); it!=key_list.end(); it++)
						m.insert( make_pair(*it, val_arr[*it]) );
		}

		//private:
		int capacity;
		double* val_arr;
		vector<int> key_list;
};

#endif
