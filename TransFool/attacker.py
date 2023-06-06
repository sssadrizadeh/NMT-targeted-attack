import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)


from datasets import load_metric
import torch
from utils.eval import cosine_distance
from utils.load import isolate_source_lang 
from utils.attack import decode, attack_successful, adv_targeted_loss, best_attack_target
from utils.attack_output import attack_output
import numpy as np




class Attacker():
    def __init__(self, args, model=None, tokenizer=None, tokenized_dataset=None, device=None, attack_type='white'): 
        self.model = model              # target NMT model
        self.model.eval()
        self.tokenizer = tokenizer      
        self.tokenized_dataset = tokenized_dataset          
        self.args = args
        self.device = device
        self.attack_type = attack_type
        self.scale = self.model.get_encoder().embed_scale





        # isolating source language token ids
        self.all_source_index = isolate_source_lang(self.tokenized_dataset)

        # Find the embeddings of all tokens
        with torch.no_grad():
            self.embeddings = self.model.get_input_embeddings()(torch.Tensor(self.all_source_index).long().to(self.device))

        self.eos_id = self.tokenizer.eos_token_id
        self.EOS_embdedding = self.model.get_input_embeddings()(torch.Tensor([self.eos_id]).long().to(self.device))[0]

        # token distance calculator
        self.cos_dist = cosine_distance(self.tokenized_dataset, self.tokenizer, self.device, mode="avg")


    # TransFool attack
    def gen_adv(self, sentence_number, LM_model, fc, attack_target=None,attack_ids_target=None,):
        
        lr, list_w_sim = self.args.lr, self.args.w_sim

        part = "test"
        metric_bleu = load_metric("sacrebleu")
        metric_chrf = load_metric("chrf")

        

        # LM embeddings
        fc.eval()    
        LM_model.eval()
        # Find the embeddings of all tokens
        with torch.no_grad():
            # embeddings of the LM
            LM_embeddings = fc(self.embeddings)


        
            
        ref_tr = self.tokenized_dataset[part][sentence_number]['labels']         
                    
        input_ids = self.tokenized_dataset[part][sentence_number]['input_ids']
        input_sent = self.tokenizer.decode(input_ids, skip_special_tokens=True).replace("▁"," ")
        
        
        ref_tr_decode = [[decode(ref_tr,self.tokenizer,self.args.model_name,self.args.target_lang)]]

        first_tr_ids = self.model.generate(torch.LongTensor(input_ids).unsqueeze(0).to(self.device))
        first_tr = [decode(first_tr_ids[0],self.tokenizer,self.args.model_name,self.args.target_lang)]

        
        org_bleu = metric_bleu.compute(predictions=first_tr, references=ref_tr_decode)['score']
        org_chrf = metric_chrf.compute(predictions=first_tr, references=ref_tr_decode)['score']
        
        best  = attack_output('failed',input_sent, \
                    input_sent, \
                    first_tr[0], \
                    ref_tr_decode[0][0], \
                    first_tr[0], \
                    org_bleu, org_bleu, org_chrf, org_chrf,0)
        
        output_ids = first_tr_ids.detach()#self.model.generate(torch.LongTensor(input_ids).unsqueeze(0).to(self.device))
        pred_project_decode = first_tr#[decode(output_ids[0],self.tokenizer)]
        
        if attack_target!=None:
                if attack_successful(attack_target, pred_project_decode[0].lower(), attack_ids_target, output_ids[0])==len(attack_ids_target):
                    best.attack_result="no_Nth"
                    print(f'target word is already in {sentence_number}!')
                    return best
        
        
        tr_ref = ref_tr if self.args.ref_or_first=="ref" else output_ids[0]
        
        for w_sim_coef in list_w_sim:
            w = [self.EOS_embdedding if i==self.eos_id else self.embeddings[self.all_source_index.index(i)].cpu().numpy() for i in input_ids]
            w = torch.tensor(w,requires_grad=False).to(self.device)
            
            LM_w = torch.tensor(fc(w[:-1].detach()),requires_grad=False)
            
            all_w = [ torch.clone(w.detach()) ]

            w_a = torch.tensor(w,requires_grad=True).to(self.device)
            optimizer = torch.optim.Adam([w_a],lr=lr)
             
            itr = 0
            
            print(attack_target)
            

            while  itr< 500: 
                
                itr+=1  

                output = self.model(inputs_embeds=self.scale*w_a.unsqueeze(0), labels = torch.tensor(tr_ref).unsqueeze(0).to(self.device))
                lm_logits = output.logits

                if itr==1 and self.args.Nth!=None and w_sim_coef==list_w_sim[0]:
                    attack_ids_target, attack_target, best_index = best_attack_target(tr_ref, self.tokenizer, lm_logits, self.args.Nth)
                    print(best_index,attack_target)
                    if attack_target == None:
                        best.attack_result="no_Nth"
                        print(f'No target id on {sentence_number}!')
                        return best 

                
                if itr==1 and w_sim_coef==list_w_sim[0]:
                    logit_diff, mask, adv_loss = adv_targeted_loss(lm_logits, attack_ids_target, self.device)
                    best_index = mask
                    print(best_index)
                else:
                    if self.args.fix_or_flex=="flex":
                        _, mask, adv_loss = adv_targeted_loss(lm_logits, attack_ids_target, self.device)
                    else:
                        _, mask, adv_loss = adv_targeted_loss(lm_logits, attack_ids_target, self.device, mask)

                LM_input = fc(w_a[:-1]).to(self.device)
                
                
                sim_loss = self.cos_dist.calc_dist(input_ids[:-1],LM_w,LM_input)

                total_loss = adv_loss + w_sim_coef * sim_loss 

                optimizer.zero_grad()
                total_loss.backward()
                
                if self.args.model_name=="marian":
                    fix_idx = torch.from_numpy(np.array([len(input_ids)-1])).to(self.device)
                elif self.args.model_name=="mbart":
                    fix_idx = torch.from_numpy(np.array([0,len(input_ids)-1])).to(self.device)
                w_a.grad.index_fill_(0, fix_idx, 0)
                
                
                optimizer.step()
                
                print(f'itr: {itr} \t w_sim_coef: {w_sim_coef} \t loss: {adv_loss.data.item(), sim_loss.data.item(), total_loss.data.item()}') 
            
                cosine = -1 * torch.matmul(LM_input,LM_embeddings.transpose(1, 0))/torch.unsqueeze(LM_input.norm(2,1), 1)/torch.unsqueeze(LM_embeddings.norm(2,1), 0)
                
                index_prime = torch.argmin(cosine,dim=1)
                w_prime = torch.cat((self.embeddings[index_prime],self.EOS_embdedding.unsqueeze(0)),dim=0)
                index_prime = torch.tensor([self.all_source_index[index_prime[i]] if i<len(w_prime)-1 else self.eos_id for i in range(len(w_prime))],requires_grad=False)
                ref_index_prime = torch.tensor(self.tokenizer.encode(self.tokenizer.decode(index_prime,  skip_special_tokens=True)),requires_grad=False) # to account for tokenization artefacts



                adv_sent = self.tokenizer.decode(index_prime, skip_special_tokens=True).replace("▁"," ")
                print(adv_sent)

                
                if torch.equal(w_prime,w)==False:
                    skip=False
                    for w_ in all_w:
                        if torch.equal(w_prime,w_):
                            skip=True
                    if skip==False:
                        print('*****************')
                        w_a.data=w_prime
                        all_w.append(torch.clone(w_prime.detach()))
                    
                        output_ids = self.model.generate(ref_index_prime.unsqueeze(0).to(self.device))
                        pred_project_decode = [decode(output_ids[0],self.tokenizer,self.args.model_name,self.args.target_lang)]
                        
                        best.query=best.query+1
                        print(best.query)
                        
                        adv_bleu = metric_bleu.compute(predictions=pred_project_decode, references=ref_tr_decode)['score']
                        adv_chrf = metric_chrf.compute(predictions=pred_project_decode, references=ref_tr_decode)['score']

                        n_success = attack_successful(attack_target, pred_project_decode[0].lower(), attack_ids_target, output_ids[0])
                        if (n_success>best.n_success) or (adv_loss.data.item() < best.adv_loss and n_success==best.n_success):
                            best.adv_sent = adv_sent
                            best.adv_tr = pred_project_decode[0]
                            best.adv_bleu = adv_bleu
                            best.adv_chrf = adv_chrf
                            best.adv_loss = adv_loss.data.item()
                            best.attack_pos = mask
                            best.n_success = n_success

                        print('bleu score: ', adv_bleu)
                        if n_success==len(attack_ids_target):
                            best.attack_result = 'success'
                            break

                best.itr = itr
            
            if best.attack_result == 'success':
                break

        
        
        best.logit_diff = logit_diff
        best.attack_target = attack_target
        best.first_pos = best_index

        if best.attack_result == 'failed':
            print(f'Failed to generate Adv attack on {sentence_number}!')
            
        elif best.attack_result == 'success':
            print(f'Successed to generate Adv attack on {sentence_number}!')
   
        return best
            

