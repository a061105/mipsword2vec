#include "util.h"
#include "multi.h"
#include "skipgram_loss.h"
#include "exact_mips_skipgram.h"
#include "decomp_mips_skipgram.h"
#include "gd_solve.h"
#include "lbfgs_solve.h"

void exit_with_help(){
	cerr << "Usage: train (options) [train_data] (model_fname)" << endl;	
	cerr << "-l loss: (default 0)" << endl;
	cerr << "		0 --- skip-gram (exact)" << endl;
	cerr << "		1 --- skip-gram (mips-exact)" << endl;
	cerr << "		2 --- skip-gram (mips-approx)" << endl;
	cerr << "		3 --- skip-gram (decomp-mips)" << endl;
	cerr << "-s algorithm: (default 0)" << endl;
	cerr << "		0 --- gradient descent" << endl;
	cerr << "		1 --- LBFGS" << endl;
	cerr << "-v vec_dim: size of the embedding vector (default 100)" << endl;
	cerr << "-t step_size: GD initial step size (default 0.1)" << endl;
	cerr << "-f factor_dim: dimension per factor for dual-decomposed loss (default 10)" << endl;
	cerr << "-q query_size: #label retrieved from MIPS per query for dual-decomposed loss (default 10000)" << endl;
	cerr << "-c num_cluster: #cluster in MIPS for dual-decomposed loss (default 400)" << endl;
	cerr << "-r num_root_cluster: #root_cluster in hierarchical MIPS (default -1)" << endl;
	cerr << "-n num_thread: #threads for parallelization (default 10)" << endl;
	cerr << "-z sample_size: #classes sampled per update in sampled softmax loss (default 1000)" << endl;
	cerr << "-e num_epoch: number of epoches for running SGD (default 1000)" << endl;
	cerr << "-i init_model: model file used as initialization." << endl;
	exit(0);
}


void parse_cmd_line(int argc, char** argv, Param* param){

	int i;
	for(i=1;i<argc;i++){
		if( argv[i][0] != '-' )
			break;
		if( ++i >= argc )
			exit_with_help();
		
		vector<string> tokens;
		switch(argv[i-1][1]){
			
			case 'l': param->loss = atoi(argv[i]);
					break;
			case 's': param->solver = atoi(argv[i]);
					break;
			case 'v': param->vec_dim = atoi(argv[i]);
					break;
			case 'f': param->factor_dim = atoi(argv[i]);
					break;
			case 't': param->init_step_size = atof(argv[i]);
				  break;
			case 'q': param->query_size = atoi(argv[i]);
				  break;
			case 'c': param->num_cluster = atoi(argv[i]);
				  break;
			case 'r': param->num_root_cluster = atoi(argv[i]);
				  break;
			case 'n': param->num_thread = atoi(argv[i]);
					break;
			case 'z': param->sample_size = atoi(argv[i]);
					break;
			case 'e': param->num_epoch = atoi(argv[i]);
					break;
			case 'i': param->init_model = argv[i];
					break;
			default:
				  cerr << "unknown option: -" << argv[i-1][1] << endl;
				  exit(0);
		}
	}

	if(i>=argc)
		exit_with_help();
	
	param->trainFname = argv[i++];
	
	if( i < argc )
			param->modelFname = argv[i];
}


int main(int argc, char** argv){
		
		Param* param = new Param();
		parse_cmd_line(argc, argv, param);
		/*int seed = time(NULL);
		cerr << "seed=" << seed << endl;
		srand(seed);*/
		omp_set_num_threads( param->num_thread );

		//read data
		Problem* prob = new Problem();
		readData( param->trainFname, prob);
		cerr << "N=" << prob->N << ", |vocab|=" << prob->K << ", vec_size=" << param->vec_dim << endl;
		
		//model initialization
		Model* model = new Model(prob->K, param->vec_dim);
		
		//training objective
		Function* func = NULL;
		if( param->loss==0 ){
				cerr << "Skipgram" << endl;
				func = new SkipgramLoss(prob, model);

		}else if( param->loss==1 ){
				cerr << "Skipgram + MIPS (exact)" << endl;
				func = new ExactMipsSkipgram(prob, model, param->query_size);

		}else if( param->loss==2 ){

				cerr << "Skipgram + MIPS (approx)" << endl;
				double prob_discard = 0.005;
				func = new DecompMipsSkipgram(prob, model, param->query_size, param->factor_dim, param->num_cluster, prob_discard, param->num_thread );

		}else if( param->loss==3 ){
				
				cerr << "Skipgram + MIPS (decomposed)" << endl;
				//func = new MipsSkipgram(prob, model, param->query_size);
		}else{
				cerr << "unknown loss: " << param->loss << endl;
				exit(0);
		}

		//solve
		Solver* solver = NULL;
	 	if( param->solver==0 ){
				cerr << "GD solve..." << endl;
				solver = new GDSolve(param->init_step_size, param->num_epoch);
		}else if( param->solver==1 ){
				cerr << "LBFGS solve..." << endl;
				int m=10;
				double tol = 1e-6;
				solver = new LBFGSSolve( tol, param->num_epoch, m );
		}else{
				cerr << "unknown solver: " << param->solver << endl;
				exit(0);
		}
		
		solver->minimize(func, model->U, model->V);
		
		writeModel( param->modelFname, model );

		return 0;
}
