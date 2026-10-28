import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from tse.utils import load_audio

class TSEManifestDataset(Dataset):
    def __init__(self,manifest,sample_rate=16000):
        self.manifest=Path(manifest); self.sample_rate=sample_rate
        if not self.manifest.exists(): raise FileNotFoundError(self.manifest)
        self.rows=[json.loads(s) for s in self.manifest.read_text(encoding='utf-8').splitlines() if s.strip()]
        if not self.rows: raise ValueError(f'empty manifest: {manifest}')
        missing=[r['id'] for r in self.rows if not Path(r['embedding']).exists()]
        if missing: raise FileNotFoundError(f'{len(missing)} embeddings missing; run scripts/precompute_embeddings.py (example {missing[0]})')
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; mix=load_audio(r['mixture'],self.sample_rate); target=load_audio(r['target'],self.sample_rate)
        n=min(mix.numel(),target.numel()); data=torch.load(r['embedding'],map_location='cpu',weights_only=False)
        emb=data['embedding'] if isinstance(data,dict) else data
        return {'id':r['id'],'mixture':mix[:n],'target':target[:n],'embedding':emb.float(),'row':r,'task':r.get('task','unknown')}

def tse_collate(batch):
    n=min(x['mixture'].numel() for x in batch)
    return {'ids':[x['id'] for x in batch], 'mixture':torch.stack([x['mixture'][:n] for x in batch]),
            'target':torch.stack([x['target'][:n] for x in batch]), 'embedding':torch.stack([x['embedding'] for x in batch]),
            'rows':[x['row'] for x in batch], 'tasks':[x['task'] for x in batch]}
