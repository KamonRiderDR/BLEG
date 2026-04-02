import argparse


def get_args():
	parser = argparse.ArgumentParser(description='LLM4Brain')

	#* TRAINING parameters 
	parser.add_argument('--root', type=str, default="../")
	parser.add_argument('--path', type=str, default="../data/", help="path of the dataset")
	parser.add_argument('--result_path', type=str, default=None, help="path of the saved-results")
	parser.add_argument('--device', type=str, default='0')
	parser.add_argument('--seed', type=int, default=777, help='random seed')
	parser.add_argument('--dataset', type=str, default="HCP", help='name of the dataset')
	parser.add_argument('--folds', type=int, default=10, help='number of folds (default: 10)')
	parser.add_argument('--times', type=int, default=10, help='number of repetitions (default: 10)')
	parser.add_argument('--threshold', type=float, default=0.12, help='threshold')
	parser.add_argument('--epochs_pretrain', type=int, default=300)
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')

	#* MODEL parameters
	parser.add_argument('--input_dim', 		type=int,	default=90)
	parser.add_argument('--dropout', 		type=float,	default=0.0)
	parser.add_argument('--num_classes',	type=int,	default=2)
	parser.add_argument('--num_layers', 	type=int,	default=3)
	parser.add_argument('--hidden_dim', 	type=int,	default=1024)
	parser.add_argument('--lambda_kd', 		type=float,	default=0.2)

	# compareison model parameters
	parser.add_argument('--model', type=str, default="GCN", help='for comparison model only')
	parser.add_argument('--num_heads', type=int, default=4, help='number of heads for attention')
	parser.add_argument('--pos_enc', type=str, default="gcn", help='position encoding type')
	parser.add_argument('--d_k', type=int, default=64, help='dimension of key')
	parser.add_argument('--d_v', type=int, default=64, help='dimension of value')

	#* IMPORTANT parameters
	parser.add_argument('--dataset_pool', 
                     	type=list,
                      	default=["MDD", "ABIDE", "ADHD", "HCP"],
                        help="all datasets concerned")
	
	parser.add_argument('--exception',
                        type=list,
                        default=["MDD", "ABIDE", "ADHD"],
						help="datasets not used for training (exception + dataset == dataset_pool)")

	parser.add_argument('--training_type', 
                        type=str, 
                        default="pretrain",
                        choices=["pretrain", "distill", "peft", "sft", "instruct"])
								# [Options]:
								# 		pretrain: 		default type, frozen lm and train adapter, classifier
								# 		peft: 			trainer and lora for pretraining
								# 		distill:		pretraining lm via llm text (classification, QA), finetune for adapter and classifier.
								# 		instruct:		pretraining lm via llm text (description, QA)
								# 		sft: 			finetune classifier only

	parser.add_argument('--exp_type', 
                        type=str,
                        default="k_fold",
                        choices=["k_fold", "few_shot", "k_way"],
                        help="experiment setting")
						# [Options]:
						#		k-fold cross-validation
						#		few-shot (training fold)
						#		k-way few shot

	# hyper parameters for few-shot training
	parser.add_argument('--few_shot_hyper', 
                        type=float, 
                        default=0.8,
                        help="few shot parameters ")
						# [Options]:
						#		few_shot:	training: hyper * len(dataset)
						#//		k_way:		training: hyper * (num_classes)


	# hyper parameters for k-way shot
	parser.add_argument('--k_way_k', 
						type=int,
						default=5,
						help="k-way shot learning number for training")
	parser.add_argument('--k_way_q',
						type=int,
						default=-1,
						help="k-way shot learning number for testing")
						# [Options]:
						#		-1:		all the data is used for testing 
						#	   n>0:		select q number per class for testing

	# IMPORTANT TRAINING hyper parameters
	parser.add_argument('--norm', 
                        type=str, 
                        default="batch",
                        choices=["batch", "layer"])
	
	parser.add_argument('--act_fn', 
                        type=str, 
                        default="gelu",
                        choices=["relu", "gelu", "swish", "tanh"])

	parser.add_argument('--epochs', 
                        type=int, 
                        default=500, 
                        help='maximum number of epochs')
	
	parser.add_argument('--patience', 
                        type=int,
                        default=50,
                        help='patience for early stopping')

	parser.add_argument('--lr', 
                        type=float, 
                        default=0.0001, 
                        help='learning rate') # [0.01, 0.0001]

	parser.add_argument('--weight_decay', 
                        type=float, 
                        default=0.00, 
                        help='weight decay') # [0.00, 0.01]

	args = parser.parse_args()
	return args
