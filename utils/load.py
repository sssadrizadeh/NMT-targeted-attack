from transformers import GPT2LMHeadModel, GPT2Config, AutoModelForSeq2SeqLM, AutoTokenizer, MBart50TokenizerFast
from datasets import load_dataset, DatasetDict
import functools 
import torch
import numpy as np
import os

def get_ids_adv_text(len_total_ids,num_ids,start_idx,seed=113):
  # Generate the ids of the sentences to attack in the dataset
  np.random.seed(seed)
  ids=np.arange(len_total_ids)
  np.random.shuffle(ids)
  return ids[start_idx:num_ids].tolist()


def preprocess_function(examples, source_lang, target_lang, prefix, tokenizer):
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets,  truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs



def load_tokenized_dataset(tokenizer,source_lang,target_lang,dataset_name=None,dataset_config_name=None,dataset=None,part=None):
    if dataset_name and part is None:
        dataset = load_dataset(dataset_name, dataset_config_name)
        dataset['train'] = dataset['train'].select(range(min(len(dataset['train']),3000000)))
    elif dataset_name and part is not None:
        dataset = load_dataset(dataset_name, dataset_config_name,split=part)
        dataset = DatasetDict({part: dataset})
    elif dataset_name is None and dataset is None:
        print("error! no dataset")

    tokenized_dataset = dataset.map(functools.partial(preprocess_function, source_lang=source_lang, target_lang=target_lang, prefix='', tokenizer=tokenizer), batched=True) 
    return dataset , tokenized_dataset



def load_model_tokenizer(model_name, source_lang, target_lang, device):
    # load NMT model
    if model_name=="marian":
        name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    elif model_name=="mbart":
        name = 'facebook/mbart-large-50-one-to-many-mmt'
    model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device)

    # load tokenizer
    if model_name=="marian":
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    elif model_name=="mbart":
        tokenizer = MBart50TokenizerFast.from_pretrained(name)
        dict_lang = {"en":"en_XX", "de":"de_DE", "fr":"fr_XX", "zh":"zh_CN"}
        tokenizer.src_lang = dict_lang[source_lang]
        tokenizer.tgt_lang = dict_lang[target_lang]
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id[tokenizer.tgt_lang ]
        
    tokenizer.model_max_length = 512
    return model, tokenizer



def load_LM_FC(model_name, source_lang, target_lang, vocab_size, embedding_size, device):
    root = os.getcwd()+"/LanguageModel"
    
    if model_name=="marian":
        dict_path = root + f"/marian_{source_lang}_{target_lang}/" 
    elif model_name=="mbart":
        dict_path = root + "/mbart/" 
    
    # Language model
    LM_model = load_gpt2_from_dict(dict_path=dict_path+"gpt2.pth", vocab_size=vocab_size, device=device, output_hidden_states=True).to(device)    

    # Fully Connected
    fc = load_fc(dict_path=dict_path+"fc.pth", NMT_embed_size=embedding_size, device=device).to(device)

    return LM_model, fc


# Language model
def embedding_from_weights(w):
    layer = torch.nn.Embedding(w.size(0), w.size(1))
    layer.weight.data = w

    return layer

def load_gpt2_from_dict(dict_path, vocab_size, device, output_hidden_states=False):
    state_dict = torch.load(dict_path,map_location=device)

    config = GPT2Config(
        vocab_size = vocab_size,
        output_hidden_states=output_hidden_states
    )
    model = GPT2LMHeadModel(config)
    model.load_state_dict(state_dict)
    # The input embedding is not loaded automatically
    model.set_input_embeddings(embedding_from_weights(state_dict['transformer.wte.weight'].cpu()))

    return model



# Fully Connected
class Net(torch.nn.Module):

    def __init__(self,NMT_embed_size):
        super(Net, self).__init__()
        
        self.fc = torch.nn.Linear(NMT_embed_size, 768) 

    def forward(self, x):
        x = self.fc(x)
        return(x)

def load_fc(dict_path,NMT_embed_size,device):
    state_dict = torch.load(dict_path,map_location=device)
    fc = Net(NMT_embed_size)
    fc.load_state_dict(state_dict)
    
    return fc


def isolate_source_lang(tokenized_dataset):
        
    all_source_index = []
    for sentence in tokenized_dataset['train']['input_ids']:
        for index in sentence[:-1]:
            all_source_index.append(index)
            
    for sentence in tokenized_dataset['test']['input_ids']:
        for index in sentence[:-1]:
            all_source_index.append(index)
            
    for sentence in tokenized_dataset['validation']['input_ids']:
        for index in sentence[:-1]:
            all_source_index.append(index)

    all_source_index = list(set(all_source_index))
    return all_source_index


def attack_tokens(attack_target,tokenizer):
    n_attack = len(attack_target.split(' '))
    with tokenizer.as_target_tokenizer():
        t = tokenizer(attack_target,  truncation=True)['input_ids']
        attack_ids_target = [i for i in t if len(tokenizer.decode(i))>3 and i not in tokenizer.all_special_ids]
        
    if len(attack_ids_target)!= n_attack:
        print("warning, multi-token word or small token!")
    
    if (n_attack > 1):
        print("The attack has more than one token!")
    else:
        print("The attack has one token!")

    return attack_ids_target