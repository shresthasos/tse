#!/usr/bin/env python3
"""Create fixed-length synthetic target-speech mixtures and JSONL manifests."""
import argparse, json, random, hashlib
from pathlib import Path
from tqdm import tqdm
from tse.utils import load_audio, save_audio, fit_length, scale_to_snr, seed_everything
EXTS={'.wav','.flac','.ogg','.mp3'}
def collect(root):
    if not root: return []
    p=Path(root)
    if not p.exists(): raise FileNotFoundError(p)
    return sorted(x for x in p.rglob('*') if x.is_file() and x.suffix.lower() in EXTS)
def speaker(p,root):
    rel=Path(p).relative_to(Path(root)); return rel.parts[0] if len(rel.parts)>=2 else Path(p).stem
def noise_crop(path,sr,n):
    w=load_audio(path,sr)
    if w.numel()<1:return w.new_zeros(n)
    if w.numel()<n:w=w.repeat((n+w.numel()-1)//w.numel())
    start=random.randint(0,w.numel()-n) if w.numel()>n else 0
    return w[start:start+n]
def build(split,root,noises,out,count,duration,sr,seed):
    rng=random.Random(seed); random.seed(seed); paths=collect(root)
    groups={}
    for p in paths:groups.setdefault(speaker(p,root),[]).append(p)
    groups={s:ps for s,ps in groups.items() if len(ps)>=2}; spks=list(groups)
    if len(spks)<2: raise ValueError(f'{split}: need >=2 speakers, with >=2 utterances each, under {root}')
    length=int(duration*sr); base=Path(out)/split
    for d in ('mixture','target','enrollment'):(base/d).mkdir(parents=True,exist_ok=True)
    rows=[]
    for i in tqdm(range(count),desc=f'prepare {split}'):
        spk=rng.choice(spks); target_path=rng.choice(groups[spk]); enroll_path=rng.choice([p for p in groups[spk] if p!=target_path])
        target=fit_length(load_audio(target_path,sr),length,True); enroll=fit_length(load_audio(enroll_path,sr),length,True)
        if noises and rng.random()<.5:
            interferer_path=rng.choice(noises); interference=noise_crop(interferer_path,sr,length); task='speech_noise'
        else:
            other=rng.choice([s for s in spks if s!=spk]); interferer_path=rng.choice(groups[other])
            interference=fit_length(load_audio(interferer_path,sr),length,True); task='two_speakers'
        snr=rng.uniform(-5,5); mixture=target+scale_to_snr(target,interference,snr)
        peak=max(float(mixture.abs().max()),1e-8); scale=min(.99/peak,1.)
        mixture*=scale; target*=scale
        sid=f'{split}_{i:06d}_{hashlib.sha1((str(target_path)+str(enroll_path)+str(seed)).encode()).hexdigest()[:8]}'
        mixf=(base/'mixture'/f'{sid}.wav').resolve(); tgtf=(base/'target'/f'{sid}.wav').resolve(); enf=(base/'enrollment'/f'{sid}.wav').resolve()
        save_audio(mixf,mixture,sr);save_audio(tgtf,target,sr);save_audio(enf,enroll,sr)
        rows.append({'id':sid,'split':split,'task':task,'speaker_id':spk,'mixture':str(mixf),'target':str(tgtf),'enrollment':str(enf),
          'embedding':str((Path(out)/'embeddings'/f'{sid}.pt').resolve()),'target_source':str(target_path),'enrollment_source':str(enroll_path),
          'interferer_source':str(interferer_path),'snr_db':round(snr,4),'sample_rate':sr,'duration_seconds':duration})
    with open(Path(out)/f'{split}.jsonl','w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(f'{split}: {len(rows)} examples -> {Path(out)/(split+".jsonl")}')
def main():
    p=argparse.ArgumentParser();p.add_argument('--train-dir',required=True);p.add_argument('--val-dir',required=True);p.add_argument('--test-dir',required=True)
    p.add_argument('--noise-dir');p.add_argument('--out',default='data/processed');p.add_argument('--samples-per-split',type=int,default=500)
    p.add_argument('--duration',type=float,default=4.);p.add_argument('--sample-rate',type=int,default=16000);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    if a.samples_per_split<1:p.error('--samples-per-split must be >= 1')
    seed_everything(a.seed);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);noises=collect(a.noise_dir)
    print(f'Found {len(noises)} noise files; if zero, only two-speaker mixtures are generated.')
    for k,(split,root) in enumerate((('train',a.train_dir),('val',a.val_dir),('test',a.test_dir))):build(split,root,noises,out,a.samples_per_split,a.duration,a.sample_rate,a.seed+k)
if __name__=='__main__':main()
