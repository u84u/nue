import torch
import torch.nn as nn
import torch.nn.functional as F

class TrainingLosses:
    def __init__(self, lambda_ce=1.0, lambda_kd=0.5):
        self.lambda_ce = lambda_ce
        self.lambda_kd = lambda_kd
        self.ce_loss = nn.CrossEntropyLoss()

    def ntp_loss(self, logits, targets):
        # logits: (B, T, V), targets: (B, T)
        return self.ce_loss(logits.view(-1, logits.size(-1)), targets.view(-1))

    def ad_loss(self, student_logits, teacher_logits, targets, temperature=1.0):
        # Autoregressive Distillation Loss
        ntp = self.ntp_loss(student_logits, targets)
        
        # KL Divergence between distributions
        s_dist = F.log_softmax(student_logits / temperature, dim=-1)
        t_dist = F.softmax(teacher_logits / temperature, dim=-1)
        kd = F.kl_div(s_dist, t_dist, reduction='batchmean')
        
        return self.lambda_ce * ntp + self.lambda_kd * kd
