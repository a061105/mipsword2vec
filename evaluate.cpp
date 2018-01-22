#include "util.h"
#include "multi.h"
#include "skipgram_loss.h"

void exit_with_help(){
	cerr << "Usage: evaluate (options) [data] [model]" << endl;	
	cerr << "-s measure: (default 0)" << endl;
	cerr << "		0 --- skipgram (cross entropy)" << endl;
	cerr << "-n #thread: (default 10)" << endl;
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
			
			case 's': param->solver = atoi(argv[i]);
					break;
			case 'n': param->num_thread = atoi(argv[i]);
					break;
			default:
				  cerr << "unknown option: -" << argv[i-1][1] << endl;
				  exit(0);
		}
	}

	if(i+1>=argc)
		exit_with_help();
	
	param->trainFname = argv[i++];
	param->modelFname = argv[i];
}


int main(int argc, char** argv){
		
		Param* param = new Param();
		parse_cmd_line(argc, argv, param);
		
		omp_set_num_threads( param->num_thread );

		//read data
		Problem* prob = new Problem();
		readData( param->trainFname, prob);
		cerr << "N=" << prob->N << endl;
		Model* model = readModel( param->modelFname );
		cerr << "|vocab|=" << model->K << ", vec_size=" << model->R << endl;
		
		//choose measure
		/*double* U = model->U;
		double* V = model->V;
		int w1 = 338;
		int w2 = 315;
		cout << "Ans:" << inner_prod(U+w1*model->R, V+w2*model->R, model->R) << endl;
		for(int k=0;k<100;k++){
				int w3 = rand() % model->K;
				cout << inner_prod(U+w1*model->R, V+w3*model->R, model->R) << endl;
		}*/
		Function* func = new SkipgramLoss(prob, model);
		cerr << "skip-gram loss=" << func->fun() << endl;
		
		return 0;
}