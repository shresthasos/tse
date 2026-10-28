import torch
import torch.nn.functional as F

def batch_si_sdr(est,ref,eps=1e-8):
    est=est.float()-est.float().mean(dim=-1,keepdim=True); ref=ref.float()-ref.float().mean(dim=-1,keepdim=True)
    proj=(torch.sum(est*ref,dim=-1,keepdim=True)/(torch.sum(ref.square(),dim=-1,keepdim=True)+eps))*ref
    noise=est-proj
    return 10*torch.log10((proj.square().sum(dim=-1)+eps)/(noise.square().sum(dim=-1)+eps))

def enhancement_loss(est,ref,si_weight=1.,l1_weight=.1):
    score=batch_si_sdr(est,ref); l1=F.l1_loss(est,ref)
    return -si_weight*score.mean()+l1_weight*l1, {'si_sdr':score.mean().detach(),'wave_l1':l1.detach()}
