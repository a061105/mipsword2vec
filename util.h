#ifndef UTIL
#define UTIL

#include<cmath>
#include<vector>
#include<set>
#include<map>
#include<string>
#include<cstring>
#include<stdlib.h>
#include<fstream>
#include<iostream>
#include<algorithm>
#include<omp.h>
#include<unordered_map>
#include<unordered_set>
#include<time.h>
#include<tuple>
#include<cassert>
#include<limits.h>
#include<queue>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include "ArrayHash.h"
#include <random>

using namespace std;

typedef vector<double> Vector;
typedef vector<Vector> Matrix;

typedef vector<pair<int,double> > SparseVec;
typedef vector<SparseVec > SparseMat;

typedef vector<int> Labels;

typedef unordered_map<int,double> HashVec;
typedef double Float;
const int LINE_LEN = 100000000;
const int FNAME_LEN = 1000;
const int INF = INT_MAX;
const int RESERVE_SIZE = 1000;

const double TOL = 1e-6;


default_random_engine generator;
normal_distribution<double> std_normal(0.0,1.0);
double randn(){
		return std_normal(generator);
}

double randu(){
		return ((double)rand()/RAND_MAX);
}

int isFile(const char* name)
{
				DIR* directory = opendir(name);

				if(directory != NULL)
				{
								closedir(directory);
								return 0;
				}

				if(errno == ENOTDIR)
				{
								return 1;
				}

				return -1;
}


ofstream& operator<<(ofstream& fout, SparseVec& sv){
		
				int size = sv.size();
				fout.write( (char*) &size, sizeof(int) );
				for(int i=0;i<sv.size();i++){
						fout.write( (char*) &(sv[i].first), sizeof(int));
						fout.write( (char*) &(sv[i].second), sizeof(Float));
				}
				
				return fout;
}

ifstream& operator>>(ifstream& fin, SparseVec& sv){
				
				int size;
				fin.read( (char*) &size, sizeof(int) );
				sv.resize(size);
				for(int i=0;i<size;i++){
						fin.read( (char*) &(sv[i].first), sizeof(int) );
						fin.read( (char*) &(sv[i].second), sizeof(Float));
				}

				return fin;
}


#define EPS 1e-12
#define INFI 1e10
#define PermutationHash HashFunc

class ScoreComp{
	
	public:
	ScoreComp(Float* _score){
		score = _score;
	}
	bool operator()(const int& ind1, const int& ind2){
		return score[ind1] > score[ind2];
	}
	private:
	Float* score;
};

class ScoreCompAsc{
	
	public:
	ScoreCompAsc(Float* _score){
		score = _score;
	}
	bool operator()(const int& ind1, const int& ind2){
		return score[ind1] < score[ind2];
	}
	private:
	Float* score;
};

class ValueComp{
	
	public:
	bool operator()(const pair<int,Float>& p1, const pair<int,Float>& p2){
		return p1.second > p2.second;
	}
};

typedef priority_queue<int,vector<int>,ScoreComp> PQueue;

class PermutationHash{
	public:
	PermutationHash(){};
	PermutationHash(int _K){	
		K = _K;
		hashindices = new int[K];
		for (int i = 0; i < K; i++){
			hashindices[i] = i;
		}
		random_shuffle(hashindices, hashindices+K);
	}
	int get(int i){
		return hashindices[i];
	}
	~PermutationHash(){
		delete [] hashindices;
	}
	int* hashindices;
	private:
	int K;
};

vector<string> split(string str, string pattern){

	vector<string> str_split;
	size_t i=0;
	size_t index=0;
	while( index != string::npos ){

		index = str.find(pattern,i);
		str_split.push_back(str.substr(i,index-i));

		i = index+1;
	}
	
	if( str_split.back()=="" )
		str_split.pop_back();

	return str_split;
}

Float inner_prod(Float* w, SparseVec* sv){

	double sum = 0.0;
	for(SparseVec::iterator it=sv->begin(); it!=sv->end(); it++){
		sum += w[it->first]*it->second;
	}
	return sum;
}

Float inner_prod(Float* u, Float* v, int d){
	
		double sum = 0.0;
		for(int j=0;j<d;j++)
				sum += u[j]*v[j];

		return sum;
}

SparseVec denseToSparse( double* v, int d ){
		
		SparseVec sv;
		sv.resize(d);
		for(int j=0;j<d;j++)
				sv.push_back(make_pair(j,v[j]));

		return sv;
}


double prox_l1_nneg( double v, double lambda ){
	
	if( v < lambda )
		return 0.0;

	return v-lambda;
}

inline Float prox_l1( Float v, Float lambda ){
	
	double v_abs = fabs(v);
	if( v_abs > lambda ){//dismec heuristic
		if( v>0.0 )
			return v - lambda;
		else 
			return v + lambda;
	}
	
	return 0.0;
}

double norm_sq( double* v, int size ){

	double sum = 0.0;
	for(int i=0;i<size;i++){
		if( v[i] != 0.0 )
			sum += v[i]*v[i];
	}
	return sum;
}

double norm_L1( SparseVec& v ){
		
			double sum = 0.0;
			for(SparseVec::iterator it=v.begin(); it!=v.end(); it++){
					sum += fabs(it->second);
			}
			
			return sum;
}

int total_size( vector<int>* alpha, int size ){
	
	int sum = 0;
	for(int i=0;i<size;i++)
		sum += alpha[i].size();
	return sum;
}

int total_size( HashVec** w, int size ){
	
	int sum = 0;
	for(int j=0;j<size;j++)
		sum += w[j]->size();
	return sum;
}

int nnz( vector<SparseVec*>& data ){
	
	int sum =0;
	for(int i=0;i<data.size();i++){
		sum += data[i]->size();
	}
	return sum;
}

void index_transpose(vector<vector<int> >& A, int N, int K, vector<vector<int> >& B){
	
	B.resize(K);
	for(int k=0;k<K;k++)
		B[k].clear();

	for(int i=0;i<N;i++)
		for(vector<int>::iterator it=A[i].begin(); it!=A[i].end(); it++)
			B[*it].push_back(i);
}

void transpose(vector<SparseVec*>& A, int N, int D, vector<SparseVec>& B){
	
	B.resize(D);
	for(int j=0;j<D;j++)
		B[j].clear();
	for(int i=0;i<N;i++){
		for(SparseVec::iterator it=A[i]->begin(); it!=A[i]->end(); it++)
			B[it->first].push_back(make_pair(i,it->second));
	}
}

void transpose(SparseVec* A, int D, int K, SparseVec* B){
	
	for(int j=0;j<K;j++)
		B[j].clear();
	for(int i=0;i<D;i++){
		for(SparseVec::iterator it=A[i].begin(); it!=A[i].end(); it++)
			B[it->first].push_back(make_pair(i,it->second));
	}
}

void transpose(double** A, int R, int C, double** Atr){
		
		for(int i=0;i<R;i++)
				for(int j=0;j<C;j++)
						Atr[j][i] = A[i][j];
}

void size_to_displacement(int* size_arr, int len, int* disp_arr){
	
	disp_arr[0] = 0;
	for(int i=1;i<len;i++)
		disp_arr[i] = disp_arr[i-1] + size_arr[i-1];
}

void size_to_displacement(long* size_arr, int len, long* disp_arr){
	
	disp_arr[0] = 0;
	for(int i=1;i<len;i++)
		disp_arr[i] = disp_arr[i-1] + size_arr[i-1];
}

string pathToFname(char* path){
			string path_str(path);
			return path_str.substr(path_str.find_last_of("/") + 1);
}


// maintain top tK indices, stored in max_indices, where indices are sorted by x[].
// Here the situation is x(i) has just been updated, where i may or may not exist in max_indices
inline bool update_max_indices(int* max_indices, Float* x, int i, int tK){
	//max_indices should have size tK+1
	int ind = 0;
	// entry ind is empty if max_indices[ind] == -1
	while (ind < tK && max_indices[ind] != -1 && max_indices[ind] != i){
		ind++;
	}
	bool adding_new_index = true;
	if (ind < tK && max_indices[ind] == i)
		adding_new_index = false;
	max_indices[ind] = i;
	int k = 0;
	//try move to right
	while (ind < tK-1 && max_indices[ind+1] != -1 && x[max_indices[ind+1]] > x[max_indices[ind]]){
		k = max_indices[ind];
		max_indices[ind] = max_indices[ind+1];
		max_indices[++ind] = k;
	}
	//try move to left
	while (ind > 0 && x[max_indices[ind]] > x[max_indices[ind-1]]){
		k = max_indices[ind];
		max_indices[ind] = max_indices[ind-1];
		max_indices[--ind] = k;
	}
	return adding_new_index;
}


const int NUM_SORT = 350;
//min_{x,y} \|x - b\|^2 + \|y - c\|^2
// s.t. x >= 0, y >= 0
//  \|x\|_1 = \|y\|_1 = t \in [0, C]
// x,b \in R^n; y,c \in R^m
// O( (m + n) log(m+n) ), but usually dominated by complexity of computing b, c
inline void solve_bi_simplex(int n, int m, Float* b, Float* c, Float C, Float* x, Float* y){
	int* index_b = new int[n];
	int* index_c = new int[m];
	for (int i = 0; i < n; i++)
		index_b[i] = i;
	for (int j = 0; j < m; j++)
		index_c[j] = j;
	
	if( n > NUM_SORT ){
			nth_element(index_b, index_b+NUM_SORT, index_b+n, ScoreComp(b));
			sort(index_b, index_b+NUM_SORT, ScoreComp(b));
	}else{
			sort(index_b, index_b+n, ScoreComp(b));
	}
	
	sort(index_c, index_c+m, ScoreComp(c));
	Float* S_b = new Float[n];
	Float* S_c = new Float[m];
	Float* D_b = new Float[n+1];
	Float* D_c = new Float[m+1];
	Float r_b = 0.0, r_c = 0.0;
	for (int i = 0; i < n; i++){
		r_b += b[index_b[i]]*b[index_b[i]];
		if (i == 0)
			S_b[i] = b[index_b[i]];
		else
			S_b[i] = S_b[i-1] + b[index_b[i]];
		D_b[i] = S_b[i] - (i+1)*b[index_b[i]];
	}
	D_b[n] = C;
	for (int j = 0; j < m; j++){
		r_c += c[index_c[j]]*c[index_c[j]];
		if (j == 0)
			S_c[j] = c[index_c[j]];
		else
			S_c[j] = S_c[j-1] + c[index_c[j]];
		D_c[j] = S_c[j] - (j+1)*c[index_c[j]];
	}
	D_c[m] = C;
	int i = 0, j = 0;
	//update for b_{0..i-1} c_{0..j-1}
	//i,j is the indices of coordinate that we will going to include, but not now!
	Float t = 0.0;
	Float ans_t_star = 0;
	Float ans = INFI;
	int ansi = i, ansj = j;
	int lasti = 0, lastj = 0;
	do{
		lasti = i; lastj = j;
		Float l = t;
		t = min(D_b[i+1], D_c[j+1]);
		//now allowed to use 0..i, 0..j
		if (l >= C && t > C){
			break;
		}
		if (t > C) { 
			t = C;
		}
		Float t_star = ((i+1)*S_c[j] + (1+j)*S_b[i])/(i+j+2);
		//cerr << "getting t_star=" << t_star << endl;
		if (t_star < l){
			t_star = l;
		//	cerr << "truncating t_star=" << l << endl;
		}
		if (t_star > t){
			t_star = t;
		//	cerr << "truncating t_star=" << t << endl;
		}
		Float candidate = r_b + r_c + (S_b[i] - t_star)*(S_b[i] - t_star)/(i+1) + (S_c[j] - t_star)*(S_c[j] - t_star)/(j+1);
		//cerr << "candidate val=" << candidate << endl;
		if (candidate < ans){
			ans = candidate;
			ansi = i;
			ansj = j;
			ans_t_star = t_star;
		}
		while ((i + 1)< n && D_b[i+1] <= t){
			i++;
			r_b -= b[index_b[i]]*b[index_b[i]];
		}
		//cerr << "updating i to " << i << endl;
		while ((j+1) < m && D_c[j+1] <= t) {
			j++;
			r_c -= c[index_c[j]]*c[index_c[j]];
		}
		//cerr << "updating j to " << j << endl;
	} while (i != lasti || j != lastj);
	//cerr << "ansi=" << ansi << ", ansj=" << ansj << ", t_star=" << ans_t_star << endl;
	for(i = 0; i < n; i++){
		int ii = index_b[i];
		if (i <= ansi)
			x[ii] = (b[index_b[i]] + (ans_t_star - S_b[ansi])/(ansi+1));
		else
			x[ii] = 0.0;
	}
	for(j = 0; j < m; j++){
		int jj = index_c[j];
		if (j <= ansj)
			y[jj] = c[index_c[j]] + (ans_t_star - S_c[ansj])/(ansj+1);
		else
			y[jj] = 0.0;
	}

	delete[] S_b; delete[] S_c;
	delete[] index_b; delete[] index_c;
	delete[] D_b; delete[] D_c;
}


void simplex_proj( HashVec& v, SparseVec& v_proj, double S){
	
		SparseVec v2;
		//v2.reserve(v.size());
		for(HashVec::iterator it=v.begin(); it!=v.end(); it++)
				v2.push_back(make_pair(it->first,it->second));
		
		if( v2.size() <= NUM_SORT )
				sort(v2.begin(), v2.end(), ValueComp());
		else{
				nth_element(v2.begin(), v2.begin()+NUM_SORT, v2.end(), ValueComp());
				sort(v2.begin(), v2.begin()+NUM_SORT, ValueComp());
		}
		
		int d = v2.size();
		int j=0;
		double part_sum = 0.0;
		for(;j<d;j++){
				double val = v2[j].second;
				part_sum += val;
				if( val - (part_sum-S)/(j+1) <= 0.0 ){
						part_sum -= val;
						j -= 1;
						break;
				}
		}
		if( j == d ) j--;
		
		double theta = (part_sum-S)/(j+1);
		
		v_proj.clear();
		for(int r=0;r<d;r++){
				
				double val = v2[r].second-theta;
				if(  val > 0.0 )
						v_proj.push_back(make_pair(v2[r].first, val));
		}
}


void simplex_proj( ArrayHash* v, SparseVec& v_proj, double S){
	
		SparseVec v2;
		v->to_sparse_vec(v2);
		
		if( v2.size() <= NUM_SORT )
				sort(v2.begin(), v2.end(), ValueComp());
		else{
				nth_element(v2.begin(), v2.begin()+NUM_SORT, v2.end(), ValueComp());
				sort(v2.begin(), v2.begin()+NUM_SORT, ValueComp());
		}
		
		int d = v2.size();
		int j=0;
		double part_sum = 0.0;
		for(;j<d;j++){
				double val = v2[j].second;
				part_sum += val;
				if( val - (part_sum-S)/(j+1) <= 0.0 ){
						part_sum -= val;
						j -= 1;
						break;
				}
		}
		if( j == d ) j--;
		
		double theta = (part_sum-S)/(j+1);
		
		v_proj.clear();
		for(int r=0;r<d;r++){
				
				double val = v2[r].second-theta;
				if(  val > 0.0 )
						v_proj.push_back(make_pair(v2[r].first, val));
		}
}


vector<int> simplex_proj( double* v, double* v_proj, int d, double S ){
	
	vector<int> index;
	index.resize(d);
	for(int i=0;i<d;i++)
		index[i] = i;

	sort(index.begin(), index.end(), ScoreComp(v)); //descending order

	int j=0;
	double part_sum = 0.0;
	for(;j<d;j++){
		int k = index[j];
		part_sum += v[k];
		if( v[k] - (part_sum-S)/(j+1) <= 0.0 ){
			part_sum -= v[k];
			j -= 1;
			break;
		}
	}
	if( j == d ) j--;
	
	double theta = (part_sum-S)/(j+1);
	
	vector<int> nz_ind;
	for(int r=0;r<d;r++){
		
		int k = index[r];
		double val = v[k]-theta;
		if(  val <= 0.0 ){
				v_proj[k] = 0.0;
		}else{
				v_proj[k] = val;
				nz_ind.push_back(k);
		}
	}

	return nz_ind;
}


vector<int> nneg_proj( double* v, double* v_proj, int d ){
	
		vector<int> nz_index;
		for(int i=0;i<d;i++)
				if( v[i] > EPS ){
						v_proj[i] = v[i];
						nz_index.push_back(i);
				}else{
						v_proj[i] = 0.0;
				}

		return nz_index;
}


int argmax(double* v, int d){
		
		double vmax = -1e300;
		int j_max = -1;
		for(int j=0;j<d;j++)
				if( v[j] > vmax ){
						vmax = v[j];
						j_max = j;
				}

		return j_max;
}

/* Assume indices in sv are sorted
 */
void max(SparseVec& sv, int full_size, double& max_val, int& max_ind){
			
		max_val = -1e300;
		if( sv.size() < full_size )
				max_val = 0.0;
		
		int ind_not_in = 0;
		for(int r=0;r<sv.size();r++){
				if( sv[r].second > max_val ){
						max_val = sv[r].second;
						max_ind = sv[r].first;
				}
				if( sv[r].first == ind_not_in )
						ind_not_in++;
		}
		
		if( max_val == 0.0 )
				max_ind = ind_not_in;
}

void normalize(double* v, int d){
		
		double sum = 0.0;
		for(int i=0;i<d;i++)
				sum += v[i]*v[i];
		sum = sqrt(sum);

		for(int i=0;i<d;i++)
				v[i] /= sum;
}

void sparse_to_dense(SparseVec& sv, double* v, int size){
		
		for(int k=0;k<size;k++)
				v[k] = 0.0;
		for(SparseVec::iterator it=sv.begin(); it!=sv.end(); it++)
				v[it->first] = it->second;
}

int max(vector<int>& v){
		
		int max_val = -10000000;
		for(int r=0;r<v.size();r++)
				if( v[r] > max_val )
						max_val = v[r];

		return max_val;
}

vector<int> list_of_first(SparseVec& sv){
		
		vector<int> ret;
		ret.reserve(sv.size());
		for(SparseVec::iterator it=sv.begin(); it!=sv.end(); it++)
				ret.push_back( it->first );

		return ret;
}

void distinct_vec( vector<int>& vec ){
	
		sort( vec.begin(), vec.end() );
		vec.erase( unique( vec.begin(), vec.end() ), vec.end() );
}

double sigmoid(double x){
		
		return 1.0 / (1.0 + exp(-x));
}
		
void factorize( SparseVec& xi, vector<int>& begin_ind, vector<int>& end_ind, int factor_dim, int num_factor){

		begin_ind.clear();
		end_ind.clear();
		begin_ind.resize(num_factor);
		end_ind.resize(num_factor);

		int r = 0;
		begin_ind[0] = r;
		for(int f=0;f<num_factor;f++){

				while( r<xi.size() && (xi[r].first/factor_dim) == f )
						r++;

				end_ind[f] = r;
				if( f+1 < num_factor )
						begin_ind[f+1] = r;
		}
}


void vadd(double a, double* v1, double b, double* v2, int d, double* v3){

		for(int i=0;i<d;i++){
				v3[i] = a*v1[i]+b*v2[i];
		}
}

#endif
