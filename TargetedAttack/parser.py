import argparse

def parser():
    parser = argparse.ArgumentParser(description="TargetedAttack Attack")

    # Bookkeeping
    parser.add_argument("--result_folder", default="result", type=str,
        help="folder for loading trained models")

    # Data
    parser.add_argument("--dataset_name", default="wmt14", type=str,
        choices=["wmt14", "opus100"],
        help="translation dataset to use")
    parser.add_argument("--dataset_config_name", default="fr-en", type=str,
        choices=["fr-en", "de-en", "en-zh"],
        help="config of the translation dataset to use")
    parser.add_argument("--num_samples", default=100, type=int,
        help="number of samples to attack")
    parser.add_argument("--start_index", default=0, type=int,
        help="starting sample index")

    # Model  
    parser.add_argument("--model_name", default="marian", type=str,
        choices=["marian", "mbart"],
        help="model which we have its gradient: target NMT model in white-box attack")
    parser.add_argument("--source_lang", default="en", type=str,
        choices=["en"],
        help="source language")
    parser.add_argument("--target_lang", default="fr", type=str,
        choices=["fr", "de", "zh"],
        help="target language") 


    # Attack setting
    parser.add_argument("--attack_target", default="", type=str,
        help="attack tokens in target language")
    parser.add_argument("--Nth", default=None, type=int,
        help="attack tokens in target language")
    parser.add_argument("--ref_or_first", default="first", type=str,
        help="reference or first translation as reference")
    parser.add_argument("--fix_or_flex", default="flex", type=str,
        help="fix the position of the attack or flexible")
    parser.add_argument("--w_sim", default=10.0, nargs='+', type=float,
        help="similarity loss coefficient")
    parser.add_argument("--w_target", default=0, type=float,
        help="dissimilarity to target word")
    parser.add_argument("--lr", default=0.020, type=float,
        help="learning rate")
    
    
    return parser