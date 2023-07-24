# coding = utf-8
import pandas as pd
import numpy as np
import torch
import argparse
import os
import datetime
import traceback

from data_loader import load_data
from train import train_model, eval_model
from model import HADE
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
import time
from tensor_parallel import TensorParallelPreTrainedModel
# CONFIG

parser = argparse.ArgumentParser(description='')
args   = parser.parse_args()

args.device        = 0
args.MAX_LEN       = 256
args.adam_epsilon  = 1e-6
args.num_class     = 2
args.drop_out      = 0.1
args.test_size     = 0.1
args.d_transformer = 128


args.print_eval = False

'''
Mode:

Uttr --> S 
Context --> S + C
Full_dialog --> F
Context_Hierarchical --> H-Flow
HADE-> HADE
'''


args.mode         = 'Uttr'

args.VAD_tokenized_dict = '../VAD_tokenized_dict.json'

args.data = 'Friends_Persona'
hade_mode = 'Full'


# args.result_name  = args.data + '_' + args.mode + '_large.txt' 

dialog_flow_len = '1' # '0.5', '0.75'
args.result_name  = args.data + '_' + args.mode + '_large_'+hade_mode+'_'+dialog_flow_len+'.txt' 






## get vad dict
VAD_Lexicons = pd.read_csv('../data/NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt', sep='\t')
VAD_dict = {}
for r in VAD_Lexicons.iterrows():
    VAD_dict[r[1]['Word']] = [r[1]['Valence'], r[1]['Arousal'], r[1]['Dominance']]
args.VAD_dict = VAD_dict


from transformers import AutoTokenizer, AutoModelForSequenceClassification

args.BASE         = 'RoBERTa'
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)


epoch_list = [3]
args.lr = 2e-4

cnt = 0

seeds = [0]#[321, 42, 1024, 0, 1, 13, 41, 123, 456, 999] # 

if args.data == 'Friends_Persona' or args.data == 'CPED':
    personalities = ['A', 'C', 'E', 'O', 'N']
    args.batch_size = 32
    args.MAX_NUM_UTTR  = 20
else:
    personalities = ['Chandler', 'Joey','Rachel','Monica','Phoebe','Ross']
    args.batch_size    = 64
    args.MAX_NUM_UTTR  = 10

with open(args.result_name, 'w') as f:
    test_acc_total = []
    test_f1_total = []
    for personality in personalities:
        args.epochs = epoch_list[0]
        cnt += 1
        if args.data == 'Friends_Persona':
            df = pd.read_csv('../data/Friends_'+personality+'_vad_'+dialog_flow_len+'.tsv', sep='\t')
#             df = pd.read_csv('../data/Friends_'+personality+'_vad.tsv', sep='\t')
        elif args.data == 'CPED':
            df = pd.read_csv('../data/CPED_'+personality+'_VAD.tsv', sep='\t')
        else:
            df = pd.read_csv('../data/PELD_'+personality+'.tsv', sep='\t')
        print('Current training classifier for', personality, '...')
        print(df.shape)
        test_acc_all_seeds = []
        test_f1_all_seeds = []
        for seed in seeds:
            args.SEED = seed
            np.random.seed(args.SEED)
            torch.manual_seed(args.SEED)
            torch.cuda.manual_seed_all(args.SEED)

            args.model_path  = './model/' + args.mode + '_'+hade_mode+'_' + str(args.MAX_LEN) + '_' + args.BASE + '_'+ str(args.lr) +'_' + '_batch_' \
                                + str(args.batch_size) + '_personality_' + personality + '_seed_' + str(seed) +'_epoch_' + str(args.epochs) + '/'

            train_dataloader, valid_dataloader, test_dataloader, train_length = load_data(df, args, tokenizer, personality)
            
            if args.mode == 'Uttr' or args.mode == 'Full_dialog':
                '''
                We use the pre-trained models to encode the utterance 
                from the speakers for personality prediction through the classification head.
                '''
                model = RobertaForSequenceClassification.from_pretrained('roberta-base', \
                                num_labels=args.num_class).cuda(args.device)
        
            elif args.mode == 'Context' :
                '''
                We input the whole dialog into the encoder for personality prediction. 
                We indicated the utterance from the analyzed speaker and the context 
                by segment embeddings in the pre-trained models: 1 for utterances and 0 for dialog context. 
                '''

                model = RobertaForSequenceClassification.from_pretrained('roberta-base', \
                                num_labels=args.num_class).cuda(args.device)

            elif args.mode == 'HADE':
                model     = HADE.from_pretrained('roberta-base', args=args).cuda(args.device)

            model = TensorParallelPreTrainedModel(model, ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])

            print('Training Length is:', len(train_dataloader))
            starttime = datetime.datetime.now()
            training_loss, best_eval_acc = train_model(model, args, train_dataloader, valid_dataloader, train_length)
            endtime = datetime.datetime.now()
            print('Training time for ', personality, ' is: ', (endtime - starttime))
            
            if args.mode == 'Uttr' or args.mode == 'Full_dialog':
                '''
                We use the pre-trained models to encode the utterance 
                from the speakers for personality prediction through the classification head.
                '''
                try:
                    model = RobertaForSequenceClassification.from_pretrained(args.model_path, \
                                    num_labels=args.num_class).cuda(args.device)
                except:
                    print(traceback.print_exc())# load the origin model

            elif args.mode == 'Context':
                try:
                    model = RobertaForSequenceClassification.from_pretrained(args.model_path, \
                                    num_labels=args.num_class).cuda(args.device)
                except:
                    print(traceback.print_exc())# load the origin model

            elif args.mode == 'HADE':
                try:
                    model     = HADE.from_pretrained(args.model_path, args=args).cuda(args.device)
                except:
                    print(traceback.print_exc())# load the origin model
            

            print('Load model from', args.model_path)
            print('Test Length is:', len(test_dataloader))
            starttime = datetime.datetime.now()
            args.print_eval = True
            test_acc, test_f1 = eval_model(model, args, test_dataloader)
            # modify here f1
            endtime = datetime.datetime.now()
            print('Inference time for ', personality, ' is: ', (endtime - starttime))
            test_acc_all_seeds.append(test_acc)
            test_f1_all_seeds.append(test_f1)
            print('Current Seed is', seed)
            print('Test ACC:', test_acc)
            print('Test F1:', test_f1)
            print('*'* 10, test_f1_total)
            print('*'* 10, test_acc_total)
            print()
            
        test_acc_total.append(test_acc_all_seeds)
        test_f1_total.append(test_f1_all_seeds)
        print('\n========\n')
        print('ACC:', test_acc_total)
        print('F1:', test_f1_total)
    f.write(str(test_acc_total))
    f.write(str(test_f1_total))


