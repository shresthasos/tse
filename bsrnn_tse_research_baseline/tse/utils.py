import json, platform, subprocess, random
from pathlib import Path
import numpy as np
import torch, torchaudio

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def choose_device(value='auto'):
    if value == 'auto': return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if value == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable; enable a GPU runtime or use --device cpu.')
    return torch.device(value)

def load_audio(path, sr=16000):
    wav, old_sr = torchaudio.load(str(path))
    if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
    if old_sr != sr: wav = torchaudio.functional.resample(wav, old_sr, sr)
    return wav.squeeze(0).float()

def save_audio(path, wav, sr=16000):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav.detach().float().cpu().reshape(1,-1).clamp(-1,1), sr)

def fit_length(wav, length, random_crop=False):
    if wav.numel() >= length:
        start=random.randint(0,wav.numel()-length) if random_crop else (wav.numel()-length)//2
        return wav[start:start+length]
    if wav.numel()==0: return torch.zeros(length)
    return wav.repeat((length+wav.numel()-1)//wav.numel())[:length]

def rms(w): return torch.sqrt(w.square().mean()+1e-8)

def scale_to_snr(signal, interference, snr_db):
    return interference * (rms(signal)/(10**(snr_db/20))/(rms(interference)+1e-8))

def si_sdr(est, ref, eps=1e-8):
    n=min(est.numel(),ref.numel()); est=est[:n].float(); ref=ref[:n].float()
    est=est-est.mean(); ref=ref-ref.mean()
    proj=(torch.sum(est*ref)/(torch.sum(ref.square())+eps))*ref
    noise=est-proj
    return (10*torch.log10((proj.square().sum()+eps)/(noise.square().sum()+eps))).item()

def write_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False)

def environment_info():
    info={'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,'torchaudio':torchaudio.__version__,'numpy':np.__version__,'cuda_available':torch.cuda.is_available()}
    if torch.cuda.is_available(): info['gpu']=torch.cuda.get_device_name(0)
    try: info['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except Exception: info['git_commit']=None
    return info
