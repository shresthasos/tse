#!/usr/bin/env python3
"""Extract normalized ECAPA-TDNN embeddings from enrollment WAVs."""
import argparse,json
from pathlib import Path
import torch
from tqdm import tqdm
from tse.utils import load_audio,choose_device

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',default='data/processed');p.add_argument('--device',default='auto');p.add_argument('--model-id',default='speechbrain/spkrec-ecapa-voxceleb');a=p.parse_args()
 dev=choose_device(a.device)
 try:from speechbrain.inference.speaker import EncoderClassifier
 except ImportError:from speechbrain.pretrained import EncoderClassifier
 encoder=EncoderClassifier.from_hparams(source=a.model_id,run_opts={'device':str(dev)})
 root=Path(a.data_root)
 for split in ('train','val','test'):
  mf=root/f'{split}.jsonl'
  if not mf.exists():raise FileNotFoundError(mf)
  rows=[json.loads(x) for x in mf.read_text().splitlines() if x.strip()]
  for row in tqdm(rows,desc=f'ECAPA {split}'):
   audio=load_audio(row['enrollment'],16000).to(dev).unsqueeze(0)
   with torch.no_grad():emb=encoder.encode_batch(audio).reshape(-1).detach().cpu().float()
   emb=torch.nn.functional.normalize(emb,dim=0); dest=Path(row['embedding']);dest.parent.mkdir(parents=True,exist_ok=True)
   torch.save({'embedding':emb,'model_id':a.model_id,'sample_id':row['id']},dest)
 print('Embeddings saved. Keep model ID and dependency versions with experiment metadata.')
if __name__=='__main__':main()
