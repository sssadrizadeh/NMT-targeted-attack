
import os
from datasets import load_metric
import torch 
from utils.eval import wer
from utils.attack_output import attack_output
from utils.load import preprocess_function
import math
from torch.nn import CrossEntropyLoss
from nltk import word_tokenize


# tokeniser decode only for zh is different
def decode(ids,tokenizer, model_name, target_lang):
        if model_name=="mbart" and target_lang=="zh":
            return tokenizer.decode(ids, skip_special_tokens=True).replace(" ","")
        else:
            return tokenizer.decode(ids, skip_special_tokens=True)

def save_outputs(args, best, attack_type, attack_alg="TargetedAttack"):

    model_name = f'white_box/{args.model_name}'
    PATH = f'{attack_alg}/{args.result_folder}/{model_name}_{args.source_lang}_{args.target_lang}'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # write original sentence
    with open(f'{PATH}/original_sentences', 'a') as f:
        f.write("%s\n" % best.org_sent)
    
    # write adversarial sentence
    with open(f'{PATH}/adversarial_sentences', 'a') as f:
        f.write("%s\n" % best.adv_sent)
    
    # write original translation
    with open(f'{PATH}/original_translations', 'a') as f:
        f.write("%s\n" % best.org_tr)
    
    # write adversarial translation
    with open(f'{PATH}/adversarial_translations', 'a') as f:
        f.write("%s\n" % best.adv_tr)
    
    # write true translation
    with open(f'{PATH}/true_translations', 'a') as f:
        f.write("%s\n" % best.ref_tr)

    # write original bleu socre
    with open(f'{PATH}/org_bleu_socre', 'a') as f:
        f.write("%s\n" % best.org_bleu)

    # write adversarial bleu socre
    with open(f'{PATH}/adv_bleu_socre', 'a') as f:
        f.write("%s\n" % best.adv_bleu)

    # write original cHRF
    with open(f'{PATH}/org_chrf', 'a') as f:
        f.write("%s\n" % best.org_chrf)
        
    # write adversarial cHRF
    with open(f'{PATH}/adv_chrf', 'a') as f:
        f.write("%s\n" % best.adv_chrf)



def save_outputs_from_pickle(d, attack_type, attack_alg="TargetedAttack", model_name=None, black_model_name=None, result_folder=None, source_lang=None, target_lang=None, black_target_lang=None):

    model_name = f'white_box/{model_name}'
    PATH = f'{attack_alg}/{result_folder}/{model_name}_{source_lang}_{target_lang}'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # write original sentence
    with open(f'{PATH}/original_sentences', 'a') as f:
        for i in d.keys():
            f.write("%s\n" % d[i].org_sent)
    
    # write adversarial sentence
    with open(f'{PATH}/adversarial_sentences', 'a') as f:
        for i in d.keys():
            f.write("%s\n" % d[i].adv_sent)
    
    # write original translation
    with open(f'{PATH}/original_translations', 'a') as f:
        for i in d.keys():
            f.write("%s\n" % d[i].org_tr)
    
    # write adversarial translation
    with open(f'{PATH}/adversarial_translations', 'a') as f:
        for i in d.keys():
            f.write("%s\n" % d[i].adv_tr)
    
    # write true translation
    with open(f'{PATH}/true_translations', 'a') as f:
        for i in d.keys():
            f.write("%s\n" % d[i].ref_tr)

    # write original bleu socre
    with open(f'{PATH}/org_bleu_socre', 'a') as f:
        for i in d.keys():
            f.write("%s\n" % d[i].org_bleu)

    # write adversarial bleu socre
    with open(f'{PATH}/adv_bleu_socre', 'a') as f:
        for i in d.keys():
            f.write("%s\n" % d[i].adv_bleu)

    # write original cHRF
    with open(f'{PATH}/org_chrf', 'a') as f:
        for i in d.keys():
            f.write("%s\n" % d[i].org_chrf)
        
    # write adversarial cHRF
    with open(f'{PATH}/adv_chrf', 'a') as f:
        for i in d.keys():
            f.write("%s\n" % d[i].adv_chrf)




def character_difference (target_word, found_word):
    longer = ""
    shorter = ""
    if (len(target_word) > len(found_word)):
        longer = target_word.strip()
        shorter = found_word.strip()
    else:
        longer = found_word.strip()
        shorter = target_word.strip()
        
    return len(longer.replace(shorter, ''))


def attack_successful1(attack_target, generated_sent):
    threshold = 2
    
    n_success = 0
    for target_word in attack_target.split(' '): 
        for ref_word in generated_sent.split(' '):
            l = target_word if len(target_word) > len(ref_word) else ref_word
            if (character_difference(target_word, ref_word) <= min(threshold, math.ceil(0.25 * len(l)))):
                n_success+=1
                break
    return n_success


def lcs(X, Y, i, j, count):
 
    if (i == 0 or j == 0):
        return count
 
    if (X[i - 1] == Y[j - 1]):
        count = lcs(X, Y, i - 1, j - 1, count + 1)
 
    count = max(count, max(lcs(X, Y, i, j - 1, 0),
                           lcs(X, Y, i - 1, j, 0)))
 
    return count

def attack_successful(attack_target, generated_sent, attack_ids_target, ids):
    threshold = 2
    
    n_success = 0

    for i,target_word in enumerate(attack_target.split(' ')): 
        if attack_ids_target[i] in ids:
            n_success+=1
        
    return n_success


def best_attack_target(ids, tokenizer, lm_logits, kth):
    
    threshold = 3
    ref = tokenizer.decode(ids, skip_special_tokens=True)

    ref_decoded = {j:tokenizer.decode(ref_token) for j,ref_token in enumerate(ids) if (ref_token not in tokenizer.all_special_ids and len(tokenizer.decode(ref_token)) > threshold)}
    indices = list(ref_decoded.keys())
    if len(ref_decoded)==0:
        return None, None, None
        
    k_th = torch.topk(lm_logits[0][indices] , kth)
    diffs = lm_logits[0][indices].max(dim=1)[0] - k_th[0][:,-1]

    
    candidate_decode = {indices[i]:(index,tokenizer.decode(index)) for i,index in enumerate(k_th[1][:,-1])}
    
    diffs = {indices[i]:diff for i,diff in enumerate(diffs) if \
             len(candidate_decode[indices[i]][1]) > threshold and \
             attack_successful(candidate_decode[indices[i]][1].lower(), tokenizer.decode(ids,skip_special_tokens=True).lower(), [candidate_decode[indices[i]][0]],ids)==0 and \
             ref_decoded[indices[i]] in word_tokenize(ref)}

    
    if len(diffs)>0:
        best_index = min(diffs, key=diffs.get)
        best_decoded = candidate_decode[best_index][1]
        attack_ids = candidate_decode[best_index][0]
    else:
        return None, None, None
    
    return [int(attack_ids)], best_decoded, best_index



def adv_targeted_loss(lm_logits, attack_ids_target, device,mask=None):
    
    adv_loss = 0

    logit_diff = []
    loss_fct = CrossEntropyLoss()

    max_logit = lm_logits.max(dim=-1)

    print("!!!!!!!!",mask)
    if mask==None:
        mask = []
        # loop through each word of attack_target and calculate the adv_loss
        for i,attack_id in enumerate(attack_ids_target):
        
            best_poss = torch.topk(-max_logit.values[0] + lm_logits[0,:,attack_id],1+i)
            for k, j in enumerate(best_poss.indices):
                if j not in mask:
                    best_pos = j  
                    mask.append(int(best_pos))
                    logit_diff.append(-1*best_poss.values[k].data.item())
                    break
        
        
    for i,attack_id in enumerate(attack_ids_target):            
        # get loss only for the target word using only the target word token ids 
        NMT_masked_lm_loss = loss_fct(lm_logits[:,mask[i],:], torch.tensor([attack_id],dtype=torch.long).to(device))

        adv_loss += NMT_masked_lm_loss

    
    adv_loss = adv_loss / len(attack_ids_target)  #Normalization

    return logit_diff, mask, adv_loss
