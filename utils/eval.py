import jiwer
import torch
from bert_score.utils import get_idf_dict
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
from tqdm import tqdm
from datasets import load_metric
from utils.load import load_model_tokenizer
import logging
import os


# Word Error Rate
def wer(x, y):
    x = " ".join(["%d" % i for i in x])
    y = " ".join(["%d" % i for i in y])

    return jiwer.wer(x, y)



# Cosine similarity constraint
class cosine_distance():
    
    def __init__(self, dataset, tokenizer, device, mode="avg"):
        self.mode = mode
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        
        if self.mode=="idf":
            self.idf_dict = self.calc_idf_dict()
        
    def calc_idf_dict(self):
        print("*********** calculating idf weights ***********")
        return get_idf_dict([ex["en"] for ex in self.dataset["train"]["translation"]], self.tokenizer, nthreads=20)
        
    
    def calc_dist(self,input_ids,x1,x2):
    
        x1 = x1 / torch.unsqueeze(x1.norm(2,1), 1)
        x2 = x2 / torch.unsqueeze(x2.norm(2,1), 1)

        dists = 1-(x1*x2).sum(1)

        if self.mode=="avg":
            dists = dists/dists.size(0)
        else:   
            weights =  torch.FloatTensor([self.idf_dict[idx] for idx in input_ids]).to(self.device)
            weights = weights/weights.sum()
            dists = dists*weights

        return dists.sum()


# Universal Sentence Encoder
class USE:
    def __init__(self):
        self.encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

    def compute_sim(self, clean_texts, adv_texts):
        clean_embeddings = self.encoder(clean_texts)
        adv_embeddings = self.encoder(adv_texts)
        cosine_sim = tf.reduce_mean(tf.reduce_sum(clean_embeddings * adv_embeddings, axis=1))

        return float(cosine_sim.numpy())

# Compute Token Error Rate 
class eval_TER:
    def __init__(self, device,args):
        logging.basicConfig(level=logging.ERROR)
        self.device = device
        _, self.tokenizer = load_model_tokenizer(args.target_model_name, args.source_lang, args.target_lang, device)
        
    def compute_TER(self,org_sent,adv_sent):
        input_ids = self.tokenizer(org_sent, truncation=True)["input_ids"]
        adv_ids = self.tokenizer(adv_sent, truncation=True)["input_ids"]

        return wer(adv_ids,input_ids)


class Eval:
    def __init__(self, attack_output, device, args):

        self.d = attack_output
        self.device = device
        self.args = args
        self.part = 'success'

        self.use = USE()
        self.eval_TER = eval_TER(device,args)

        self.indices = list(self.d.keys())

        self.condition = {idx:self.d[idx].org_bleu!=0 and self.d[idx].attack_result=='success' for idx in self.indices}

        self.n_sent = len([idx for idx in (self.indices) if self.d[idx].org_bleu!=0 and self.d[idx].attack_result!='no_Nth'])
        self.fail = len([idx for idx in (self.indices) if self.d[idx].org_bleu!=0 and self.d[idx].attack_result=='failed'])

        self.sim_perp_calc()

        self.results_success = self.eval_calc()


    def save_sim(self):

        PATH = f'{self.args.attack_alg}/{self.args.result_folder}/white_box/{self.args.target_model_name}_{self.args.source_lang}_{self.args.target_lang}'
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        # write adversarial similarity
        with open(f'{PATH}/sim_tr', 'w') as f:
            for item in self.sim_tr.values():
                f.write("%s\n" % item)

        # write original similarity
        with open(f'{PATH}/adv_sim', 'w') as f:
            for item in self.sim.values():
                f.write("%s\n" % item)

            
        
    def success_calc(self):
        self.condition = {idx:self.condition[idx] and self.d[idx].attack_result=='success' and self.sim[idx]>=self.args.bad_sim for idx in (self.indices)}
        
        self.sim = {idx:self.sim[idx] for idx in (self.indices) if self.condition[idx]}        
        self.TER = {idx:self.TER[idx] for idx in (self.indices) if self.condition[idx]}
           
    def sim_perp_calc(self):
        print("computing similarities ...")
        self.sim = {idx:self.use.compute_sim([self.d[idx].org_sent], [self.d[idx].adv_sent]) for idx in tqdm(self.indices) if self.condition[idx]}
        self.sim_tr = {idx:self.use.compute_sim([self.d[idx].org_tr], [self.d[idx].adv_tr]) for idx in tqdm(self.indices) if self.condition[idx]}
        
        self.use = 0
        self.TER = {idx:self.eval_TER.compute_TER(self.d[idx].org_sent,self.d[idx].adv_sent) for idx in (self.indices) if self.condition[idx]}    

        self.bad = len([idx for idx in (self.indices) if self.condition[idx] and (self.sim[idx]<self.args.bad_sim) and self.d[idx].attack_result=='success'])
        self.success_calc()

   
    def eval_tr(self, adv_tr, ref_tr, org_tr):
        bleu = load_metric("sacrebleu")

        bleu_corp_adv = bleu.compute(predictions=adv_tr, references=ref_tr)['score']
        bleu_corp_org = bleu.compute(predictions=org_tr, references=ref_tr)['score']

        return bleu_corp_adv, bleu_corp_org


    def eval_calc(self):
        
        adv_tr = [self.d[idx].adv_tr for idx in (self.indices) if self.condition[idx]]
        ref_tr = [[self.d[idx].ref_tr] for idx in (self.indices) if self.condition[idx]]
        org_tr = [self.d[idx].org_tr for idx in (self.indices) if self.condition[idx]]

        org_bleus = [self.d[idx].org_bleu for idx in (self.indices) if self.condition[idx]]
        adv_bleus = [self.d[idx].adv_bleu for idx in (self.indices) if self.condition[idx]]

        org_bleus = np.array(org_bleus)  
        adv_bleus = np.array(adv_bleus)


        bleu_corp_adv, bleu_corp_org = self.eval_tr(adv_tr, ref_tr, org_tr)

        results = []

        results.append(["Bad attacks due to low similarity/perplexity:", self.bad])
        results.append(["Failed attacks:", self.fail])
        results.append(["Attack success rate (%):", (1-(self.fail+self.bad)/self.n_sent)*100])
        results.append(["Average semantic similarity:", sum(self.sim.values()) / len(self.sim)])
        results.append(["Token error rate (%):", 100*sum(self.TER.values())/len(self.sim)])
        results.append(["Corpus adv bleu score:", bleu_corp_adv])
        results.append(["Corpus org bleu score:", bleu_corp_org])
        results.append(["Relative decrease of corpus bleu score:", (bleu_corp_org-bleu_corp_adv)/bleu_corp_org])

        return results











