import pickle
import torch
import argparse    
from utils.eval import Eval 
from utils.attack import attack_output
from tabulate import tabulate
# from TransFool import attacker
# from kNN import attacker

import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):

    if "TransFool" in args.attack_alg:
        PATH = f'{args.attack_alg}/{args.result_folder}/{args.attack_type}/{args.target_model_name}_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.start_index+args.num_samples}_sim_{"_".join(map(str, [k*10 for k in args.w_sim]))}_lr_{args.lr[0]*1000}_n_{args.Nth}_target_{args.attack_target}.pkl'
    
    elif "Seq2Sick" in args.attack_alg:
        PATH = f'{args.attack_alg}/{args.result_folder}/{args.attack_type}/{args.target_model_name}_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.start_index+args.num_samples}_const_{args.const}_lr_{"_".join(map(str, args.lr))}_itr_200_n_{args.Nth}_target_{args.attack_target}.pkl'
    
    
    
    with (open(PATH, "rb")) as f:
        d = (pickle.load(f))

    E = Eval(d, device, args)
    results = E.results_success
    

    # pdb.set_trace()


    print("\n\n")
    print("*********** RESULTS ***********")
    print("\n")               

    print("\n")

    print("In Successful Attacks:")
    print(tabulate(results, tablefmt='psql', showindex=False, numalign="left", floatfmt=".8f"))


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation.")

    # Bookkeeping
    parser.add_argument("--result_folder", default="result", type=str,
        help="folder for loading trained models")

    # Data
    parser.add_argument("--num_samples", default=100, type=int,
        help="number of samples to attack")
    parser.add_argument("--attack_alg", default="TransFool", type=str,
        choices=["TransFool", "Seq2Sick"],
        help="attack method to load reasults from corresponding folder")
    parser.add_argument("--attack_type", default="white_box", type=str,
        choices=["white_box"],
        help="attack type to load reasults from corresponding folder")

    # Model
    parser.add_argument("--attack_target", default="", type=str,
        help="attack tokens in target language")
    parser.add_argument("--Nth", default=None, type=int,
        help="attack tokens in target language")
    parser.add_argument("--target_model_name", default="marian", type=str,
        choices=["marian", "mbart"],
        help="target NMT model")
    parser.add_argument("--source_lang", default="en", type=str,
        choices=["en", "fr"],
        help="source language")
    parser.add_argument("--target_lang", default="fr", type=str,
        choices=["fr", "de", "zh", "en"],
        help="target language")
    # Eval setup
    parser.add_argument("--bad_sim", default=0, type=float,
        help="threshold for the bad similaroty")

    # Attack setting
    parser.add_argument("--start_index", default=0, type=int,
        help="starting sample index")

    parser.add_argument("--w_sim", default=15.0, nargs='+',type=float,
        help="similarity loss coefficient")
    
    
    parser.add_argument("--lr", default=0.016, nargs='+', type=float,
        help="learning rate")

    parser.add_argument("--weights", default="", type=str,
        help="weighted attack")

    parser.add_argument("--bleu", default=0.5, type=float,
        help="bleu score ratio for success")

    parser.add_argument("--const", default=1, type=float,
        help="const in seq2sick")

    parser.add_argument("--max-swap", default=1, type=int,
        help="max-swap in knn")

    args = parser.parse_args()
    main(args)