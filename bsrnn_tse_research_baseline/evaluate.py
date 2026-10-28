#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
from tqdm import tqdm
from tse.dataset import TSEManifestDataset,tse_collate
from tse.model import BSRNNTargetEnhancer
from tse.utils import choose_device,load_audio,save_audio,si_sdr,write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output-dir',default='runs/evaluation');p.add_argument('--device',default='auto');p.add_argument('--batch-size',type=int,default=1);p.add_argument('--num-workers',type=int,default=0);p.add_argument('--speaker-metrics',action='store_true');p.add_argument('--perceptual-metrics',action='store_true');p.add_argument('--ecapa-model-id',default='speechbrain/spkrec-ecapa-voxceleb');a=p.parse_args()
 dev=choose_device(a.device);out=Path(a.output_dir);enhdir=out/'enhanced';enhdir.mkdir(parents=True,exist_ok=True);ck=torch.load(a.checkpoint,map_location=dev,weights_only=False);cfg=ck.get('config',{})
 model=BSRNNTargetEnhancer(n_fft=cfg.get('n_fft',512),hop_length=cfg.get('hop_length',128),num_bands=cfg.get('num_bands',8),hidden_size=cfg.get('hidden_size',96),embedding_dim=cfg.get('embedding_dim',192)).to(dev);model.load_state_dict(ck['model']);model.eval()
 ds=TSEManifestDataset(a.manifest);dl=torch.utils.data.DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,collate_fn=tse_collate)
 encoder=None;stoi_fn=pesq_fn=None
 if a.speaker_metrics:
  try:from speechbrain.inference.speaker import EncoderClassifier
  except ImportError:from speechbrain.pretrained import EncoderClassifier
  encoder=EncoderClassifier.from_hparams(source=a.ecapa_model_id,run_opts={'device':str(dev)})
 if a.perceptual_metrics:
  try:from pystoi import stoi as stoi_fn;from pesq import pesq as pesq_fn
  except ImportError as e:raise RuntimeError('Install requirements-optional.txt to use --perceptual-metrics') from e
 def embed(path):
  w=load_audio(path,16000).to(dev).unsqueeze(0)
  with torch.no_grad():return torch.nn.functional.normalize(encoder.encode_batch(w).reshape(-1),dim=0)
 rows=[]
 for b in tqdm(dl,desc='Evaluate'):
  with torch.no_grad():pred=model(b['mixture'].to(dev),b['embedding'].to(dev)).cpu()
  for i,sid in enumerate(b['ids']):
   meta=b['rows'][i];est=pred[i];mix=b['mixture'][i];tgt=b['target'][i];n=min(est.numel(),mix.numel(),tgt.numel());est,mix,tgt=est[:n],mix[:n],tgt[:n]
   m=si_sdr(mix,tgt);e=si_sdr(est,tgt);dest=enhdir/f'{sid}.wav';save_audio(dest,est,16000)
   r={'id':sid,'task':meta.get('task','unknown'),'speaker_id':meta.get('speaker_id',''),'snr_db':meta.get('snr_db'),'mixture_si_sdr_db':m,'enhanced_si_sdr_db':e,'si_sdri_db':e-m}
   if encoder is not None:
    ee=embed(dest);te=embed(meta['target']);en=embed(meta['enrollment']);r['target_embedding_cosine']=torch.nn.functional.cosine_similarity(ee,te,dim=0).item();r['enrollment_embedding_cosine']=torch.nn.functional.cosine_similarity(ee,en,dim=0).item()
   if a.perceptual_metrics:
    from scipy.signal import resample_poly
    x=est.numpy();y=tgt.numpy();r['stoi']=float(stoi_fn(y,x,16000,extended=False))
    try:r['pesq_wb']=float(pesq_fn(16000,y,x,'wb'))
    except Exception:r['pesq_wb']=float('nan')
   rows.append(r)
 if not rows:raise RuntimeError('No evaluation rows found')
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with open(out/'metrics_per_utterance.csv','w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 sdri=np.array([r['si_sdri_db'] for r in rows]);summary={'manifest':str(Path(a.manifest).resolve()),'checkpoint':str(Path(a.checkpoint).resolve()),'num_examples':len(rows),'si_sdri_db':{'mean':float(sdri.mean()),'median':float(np.median(sdri)),'std':float(sdri.std()),'min':float(sdri.min()),'max':float(sdri.max()),'improved_examples':int((sdri>0).sum()),'improved_fraction':float((sdri>0).mean())},'metrics':{}}
 for key in fields:
  vals=[]
  for r in rows:
   try:
    v=float(r[key])
    if np.isfinite(v):vals.append(v)
   except (ValueError,TypeError):pass
  if vals:summary['metrics'][key]={'mean':float(np.mean(vals)),'median':float(np.median(vals)),'std':float(np.std(vals)),'min':float(np.min(vals)),'max':float(np.max(vals))}
 write_json(out/'metrics_summary.json',summary);print(json.dumps(summary,indent=2));print('Saved metrics and enhanced WAVs to',out)
if __name__=='__main__':main()
