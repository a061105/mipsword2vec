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
	int solver;

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
		
		loss = 0;
		solver = 0;

		train = NULL;
		vec_dim = 100;
		//solver
		init_step_size = 1.0;
		num_thread = 10;
		num_epoch = 1000;
		//for dual-decomposed loss
		factor_dim = 10;
		query_size = 10000;
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
						U[i] = randn()*5e-1;
						V[i] = randn()*5e-1;
				}
		}
		
		double* U;
		double* V;
		int K;
		int R;
};

Model* readModel(char* file){
		
		char* tmp = new char[LINE_LEN];
		
		ifstream fin(file);
		int K, R;
		fin >> K >> R;
		K -= 1;
		Model* model = new Model(K,R);
		
		fin.getline(tmp,LINE_LEN);//filter <s>
		fin.getline(tmp,LINE_LEN);//filter <s>
		for(int i=0;i<K;i++){
				
				fin >> tmp;
				
				for(int j=0;j<R;j++)
						fin >> (model->U[i*R+j]);
				for(int j=0;j<R;j++)
						fin >> (model->V[i*R+j]);
		}
		fin.close();
		delete[] tmp;

		return model;
}

void writeModel( const char* fname, Model* model ){
		
		int K = model->K;
		int R = model->R;

		ofstream fout(fname);
		fout << K+1 << " " << R << endl;
		fout << "<s>" << endl;
		
		double* U = model->U;
		double* V = model->V;
		for(int i=0;i<K;i++){
				fout << i << " ";
				for(int j=0;j<R;j++)
						fout << U[i*R+j] << " ";
				for(int j=0;j<R;j++)
						fout << V[i*R+j] << " ";
				fout << endl;
		}
		fout.close();
}

#endif
