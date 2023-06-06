class attack_output():
    def __init__(self,attack_result,adv_sent,org_sent,adv_tr,ref_tr,org_tr,org_bleu,adv_bleu,org_chrf,adv_chrf,query,itr=0,error_rate=0,logit_diff=1000000,adv_loss=1000000,attack_pos=-1,n_success=0,attack_target=""):
        self.attack_result = attack_result
        self.adv_sent=adv_sent
        self.org_sent=org_sent
        self.adv_tr=adv_tr
        self.ref_tr=ref_tr
        self.org_tr=org_tr
        self.org_bleu=org_bleu
        self.adv_bleu=adv_bleu
        self.org_chrf = org_chrf
        self.adv_chrf = adv_chrf
        self.query = query
        self.itr = itr
        self.error_rate=error_rate
        self.logit_diff = logit_diff


        self.adv_loss = adv_loss
        self.attack_pos = attack_pos
        self.n_success = n_success
        self.attack_target = attack_target
        