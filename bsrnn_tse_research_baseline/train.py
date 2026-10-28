#!/usr/bin/env python3
import argparse,csv,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tse.dataset import TSEManifestDataset,tse_collate
from tse.model import BSRNNTargetEnhancer
from tse.losses import enhancement_loss,batch_si_sdr
from tse.utils import choose_device,seed_everything,write_json,environment_info

def epoch_run(model,loader,device,optimizer=None,siw=1.,l1w=.1,clip=5.):
 train=optimizer is not None;model.train(train);tot={k:0. for k in ('loss','si_sdr','mixture_si_sdr','si_sdri','wave_l1')};n=0
 for b in tqdm(loader,leave=False):
  mix=b['mixture'].to(device);tgt=b['target'].to(device);emb=b['embedding'].to(device)
  if train:optimizer.zero_grad(set_to_none=True)
  with torch.set_grad_enabled(train):
   est=model(mix,emb);loss,parts=enhancement_loss(est,tgt,siw,l1w)
   if train:
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),clip);optimizer.step()
  with torch.no_grad():
   es=batch_si_sdr(est,tgt);ms=batch_si_sdr(mix,tgt);bs=mix.shape[0]
   tot['loss']+=loss.item()*bs;tot['si_sdr']+=es.sum().item();tot['mixture_si_sdr']+=ms.sum().item();tot['si_sdri']+=(es-ms).sum().item();tot['wave_l1']+=parts['wave_l1'].item()*bs;n+=bs
 return {k:v/max(n,1) for k,v in tot.items()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',default='data/processed');p.add_argument('--run-dir',default='runs/bsrrn_baseline');p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--num-workers',type=int,default=2);p.add_argument('--device',default='auto');p.add_argument('--seed',type=int,default=42);p.add_argument('--n-fft',type=int,default=512);p.add_argument('--hop-length',type=int,default=128);p.add_argument('--num-bands',type=int,default=8);p.add_argument('--hidden-size',type=int,default=96);p.add_argument('--embedding-dim',type=int,default=192);p.add_argument('--si-sdr-weight',type=float,default=1.);p.add_argument('--wave-l1-weight',type=float,default=.1);p.add_argument('--gradient-clip',type=float,default=5.);p.add_argument('--no-resume',action='store_true');a=p.parse_args()
 seed_everything(a.seed);dev=choose_device(a.device);root=Path(a.data_root);out=Path(a.run_dir);out.mkdir(parents=True,exist_ok=True)
 tr=TSEManifestDataset(root/'train.jsonl');va=TSEManifestDataset(root/'val.jsonl')
 trl=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,collate_fn=tse_collate,pin_memory=dev.type=='cuda');val=DataLoader(va,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,collate_fn=tse_collate,pin_memory=dev.type=='cuda')
 model=BSRNNTargetEnhancer(n_fft=a.n_fft,hop_length=a.hop_length,num_bands=a.num_bands,hidden_size=a.hidden_size,embedding_dim=a.embedding_dim).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
 cfg=vars(a).copy();cfg['device_used']=str(dev);cfg['model']='BSRNNTargetEnhancer';write_json(out/'config.json',cfg);write_json(out/'environment.json',environment_info())
 history=out/'history.csv';last=out/'last.pt';start=1;best=-float('inf')
 if last.exists() and not a.no_resume:
  ck=torch.load(last,map_location=dev,weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['epoch']+1;best=ck.get('best_val_si_sdri',best);print(f'Resuming from epoch {start}')
 fields=['epoch','train_loss','train_si_sdr','train_mixture_si_sdr','train_si_sdri','train_wave_l1','val_loss','val_si_sdr','val_mixture_si_sdr','val_si_sdri','val_wave_l1','lr','epoch_seconds']
 if not history.exists() or a.no_resume:
  with open(history,'w',newline='') as f:csv.DictWriter(f,fieldnames=fields).writeheader()
 for ep in range(start,a.epochs+1):
  tic=time.time();tm=epoch_run(model,trl,dev,opt,a.si_sdr_weight,a.wave_l1_weight,a.gradient_clip);vm=epoch_run(model,val,dev,None,a.si_sdr_weight,a.wave_l1_weight,a.gradient_clip)
  row={'epoch':ep,**{f'train_{k}':v for k,v in tm.items()},**{f'val_{k}':v for k,v in vm.items()},'lr':opt.param_groups[0]['lr'],'epoch_seconds':time.time()-tic}
  with open(history,'a',newline='') as f:csv.DictWriter(f,fieldnames=fields).writerow(row)
  best=max(best,vm['si_sdri']);state={'epoch':ep,'model':model.state_dict(),'optimizer':opt.state_dict(),'best_val_si_sdri':best,'config':cfg};torch.save(state,last)
  if vm['si_sdri']>=best:torch.save(state,out/'best.pt')
  write_json(out/'latest_metrics.json',row)
  print(f"Epoch {ep}/{a.epochs} | train SI-SDRi={tm['si_sdri']:.3f} dB | val SI-SDRi={vm['si_sdri']:.3f} dB | best={best:.3f} dB")
 print('Training finished:',out)
if __name__=='__main__':main()
