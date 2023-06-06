import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)


import torch
import pickle
from parser import TransFool_parser
from utils.load import load_tokenized_dataset, load_model_tokenizer, load_LM_FC, attack_tokens, get_ids_adv_text
from utils.attack import save_outputs
import time
from attacker import Attacker


def main(args):

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the NMT model and its tokenizer
    model, tokenizer = load_model_tokenizer(args.model_name, args.source_lang, args.target_lang, device)

    # load tokenized dataset 
    _, tokenized_dataset = load_tokenized_dataset(tokenizer,args.source_lang,args.target_lang,args.dataset_name,args.dataset_config_name)

    

    
    attack_target=None 
    attack_ids_target=None
    if args.attack_target == "" and args.Nth==None:
        args.Nth = 2
    elif args.attack_target != "":
        # attack_source = 'dog sits' #'hat' #"enemy" #'War' #'dog' #'cat' #"Committee" #'cat sits' #'cat dog'#'war police' #'police'
        # attack_target = 'chien est assis' #'chapeau' #"ennemi" #'guerre' #'chien' #'chat'  #"Comité" #"bombe"  #'chat est assis' #'chat chien'#'guerre police'
        # if len(args.attack_source.split(' '))!=len(args.attack_target.split(' ')):
        #     print("please give the attack tokens in both source and target languages!")
        # else:
        attack_target=args.attack_target
        attack_ids_target = attack_tokens(attack_target,tokenizer)          
        


    
    # create Attacker
    if args.LM=='no_LM':
        # args.lr = 0.040
        attacker = Attacker(args, model, tokenizer, tokenized_dataset, device, attack_target,attack_ids_target, attack_type='white_noLM')    
    else:
        attacker = Attacker(args, model, tokenizer, tokenized_dataset, device, attack_target,attack_ids_target)

    # load LM model and FC layer
    if args.LM=='fine_tune':
        LM_model, fc = load_LM_FC('fine_tune_LM_marian', args.source_lang, args.target_lang, tokenizer.vocab_size, attacker.embeddings.size()[-1], device)
    else:
        LM_model, fc = load_LM_FC(args.model_name, args.source_lang, args.target_lang, tokenizer.vocab_size, attacker.embeddings.size()[-1], device)

    ids_to_attack = get_ids_adv_text(len(tokenized_dataset['test']),args.num_samples,args.start_index)

    attack_dict = {}
    attack_type = "" if args.LM=='gpt2' else f'_{args.LM}'

    time_begin = time.time()
    for idx in [482]:#ids_to_attack:#range(args.start_index, args.start_index+args.num_samples):
     
        best_output = attacker.gen_adv(idx, LM_model, fc)
        attack_dict[idx]= best_output 
        # save_outputs(args, best_output, 'white'+attack_type, attack_alg="TransFool")


    print(f'finished attack for {args.num_samples} samples in {time.time()-time_begin} seconds!')
    print(time.time()-time_begin)
        
    
    folder_name_dict = { '':'white_box',\
                         '_fine_tune':'fine_tune',\
                         '_no_LM':'noLM'}

    os.makedirs(f'TransFool/{args.result_folder}/{folder_name_dict[attack_type]}', exist_ok=True)
    with open(f'TransFool/{args.result_folder}/{folder_name_dict[attack_type]}/{args.model_name}_{args.source_lang}_{args.target_lang}_{args.mode}_{args.start_index}_{args.start_index+args.num_samples}_sim_{args.w_sim*10}_perp_{args.w_perp*10}_lr_{args.lr*1000}_n_{args.Nth}_target_{args.attack_target}.pkl', 'wb') as f:
        pickle.dump(attack_dict, f)

if __name__ == '__main__':
    
    parser = TransFool_parser()
    args = parser.parse_args()

    main(args)
