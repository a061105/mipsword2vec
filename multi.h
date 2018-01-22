#ifndef MULTITRAIN
#define MULTITRAIN

#include "util.h"

class Problem{
	public:
	static int K;//voc size
	
	vector< pair<int,double>* > data; //(word_index, freq)
	vector<SparseVec* > labels; //lab1:fq1, lab2:fq2, ...
	int N;//number of samples
};

int Problem::K = -1;

class Param{
	public:
	char* trainFname;
	char* modelFname;
	Problem* train;
	int vec_dim;
	
	int loss;
	int method;

	//solver-specific param
	double init_step_size;
	int num_epoch;
		//for dual-decomposed loss
	int factor_dim;
	int query_size;
	int num_cluster;
	int num_root_cluster;
	int num_thread;
	//for sampled softmax
	int sample_size;
	
	char* init_model;

	Param(){
		
		train = NULL;
		vec_dim = 100;
		//solver
		init_step_size = 1.0;
		num_thread = 10;
		num_epoch = 1000;
		//for dual-decomposed loss
		factor_dim = 10;
		query_size = 500;
		num_cluster = 400;
		num_root_cluster = -1;
		//for sampled softmax
		sample_size = 1000;
		
		//model save/load
		init_model = NULL;
		modelFname = "model";
	}

	~Param(){
		delete[] trainFname;
		delete[] modelFname;
	}
};


//only used for prediction
	
void writeModel( const char* fname, Problem* prob, double* w ){
		
		ofstream fout(fname);
		fout << "vocabulary size " << prob->K << endl;
		/*
		int K = prob->K;
		//weights
		for(int j=0;j<D;j++){
				
				int nnz = 0;
				for(int k=0;k<prob->K;k++){
						if( fabs(w[j*K+k]) > 1e-4 )
							nnz++;
				}
				
				fout << nnz << " ";
				for(int k=0;k<prob->K;k++){
							if( fabs(w[j*K+k]) > 1e-4 )
									fout << k << ":" << w[j*K+k] << " ";
				}
				fout << endl;
		}
		cerr << endl;
		*/
		fout.close();
}



void readData(char* fname, Problem* prob )
{
	
	ifstream fin(fname);
	char* line = new char[LINE_LEN];
	int max_fea_ind = -1;
	int line_count = 1;
	while( !fin.eof() ){
		
		fin.getline(line, LINE_LEN);
		string line_str(line);
		
		if( line_str.length() < 2 && fin.eof() )
			break;
		size_t found = line_str.find("  ");
		while (found != string::npos){
			line_str = line_str.replace(found, 2, " ");
			found = line_str.find("  ");
		}
		found = line_str.find(", ");
		while (found != string::npos){
			line_str = line_str.replace(found, 2, ",");
			found = line_str.find(", ");
		}
		vector<string> tokens = split(line_str, " ");
		
		//get label index
		SparseVec* labels = new SparseVec();
		map<string,int>::iterator it;
		
		vector<string> subtokens = split(tokens[0], ",");
		for (vector<string>::iterator it = subtokens.begin(); it != subtokens.end(); it++){

				vector<string> iv_pair = split(*it, ":");
				int ind = atoi(iv_pair[0].c_str());
				double val = (iv_pair.size()>1)?  atof(iv_pair[1].c_str()) : 1.0;
				
				labels->push_back(make_pair( ind, val ));
				
				if( ind > max_fea_ind )
						max_fea_ind = ind;
		}
		
		pair<int,double>*  ins = new pair<int,double>();
		/////////////
		
		int freq = atoi(tokens[ tokens.size()-2].c_str());
		vector<string> kv = split(tokens[ tokens.size()-1 ],":");
		int word_ind = atoi(kv[0].c_str());
		
		ins->first = word_ind;
		ins->second = freq;
		if( word_ind > max_fea_ind )
				max_fea_ind = word_ind;
		
		prob->data.push_back(ins);
		prob->labels.push_back(labels);
		
		line_count++;
	}
	fin.close();
	
	prob->K = max(max_fea_ind+1, prob->K);
	prob->N = prob->data.size();
	
	delete[] line;
}

class Model{
		
		public:
		Model(int _K, int _R){
				
				K = _K;
				R = _R;

				U = new double[K*R];
				V = new double[K*R];
				
				for(int i=0;i<K*R;i++){
						U[i] = randn()*1e-1;
						V[i] = randn()*1e-1;
				}
		}
		
		double* U;
		double* V;
		int K;
		int R;
};

void readModel(char* file,  Model* model){
		/*
		char* tmp = new char[LINE_LEN];
		
		ifstream fin(file);
		fin >> tmp >> (model->K);
		fin >> tmp >> (model->D);
		model->w.resize(model->K);
		
		vector<string> ind_val;
		int nnz_j;
		for(int j=0;j<model->D;j++){
				fin >> nnz_j;
				for(int r=0;r<nnz_j;r++){
						fin >> tmp;
						ind_val = split(tmp,":");
						int k = atoi(ind_val[0].c_str());
						Float val = atof(ind_val[1].c_str());
						model->w[k].push_back( make_pair(j,val) );
				}
		}
		fin.close();

		delete[] tmp;*/
}

#endif
