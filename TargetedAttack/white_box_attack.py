import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)


import torch
import pickle
from parser import parser
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
        attack_target=args.attack_target
        attack_ids_target = attack_tokens(attack_target,tokenizer)          
        


    
    # create Attacker
    attacker = Attacker(args, model, tokenizer, tokenized_dataset, device)

    # load LM model and FC layer
    LM_model, fc = load_LM_FC(args.model_name, args.source_lang, args.target_lang, tokenizer.vocab_size, attacker.embeddings.size()[-1], device)

    ids_to_attack = get_ids_adv_text(len(tokenized_dataset['test']),args.num_samples,args.start_index)

    attack_dict = {}

    time_begin = time.time()
    for idx in ids_to_attack:
     
        best_output = attacker.gen_adv(idx, LM_model, fc, attack_target, attack_ids_target)
        attack_dict[idx]= best_output 
        # save_outputs(args, best_output, 'white', attack_alg="TargetedAttack")


    print(f'finished attack for {args.num_samples} samples in {time.time()-time_begin} seconds!')
    print(time.time()-time_begin)

    os.makedirs(f'TargetedAttack/{args.result_folder}/white_box', exist_ok=True)
    with open(f'TargetedAttack/{args.result_folder}/white_box/{args.model_name}_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.start_index+args.num_samples}_sim_{"_".join(map(str, [k*10 for k in args.w_sim]))}_lr_{args.lr*1000}_n_{args.Nth}_target_{args.attack_target}.pkl', 'wb') as f:
        pickle.dump(attack_dict, f)

if __name__ == '__main__':
    
    parser = parser()
    args = parser.parse_args()

    main(args)
