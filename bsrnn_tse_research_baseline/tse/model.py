import torch
from torch import nn

class BSRNNTargetEnhancer(nn.Module):
    """Compact BSRNN-style magnitude-mask enhancer with ECAPA embedding FiLM conditioning."""
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=128, num_bands=8, hidden_size=96, embedding_dim=192):
        super().__init__(); self.sample_rate=sample_rate; self.n_fft=n_fft; self.hop_length=hop_length
        self.num_bands=num_bands; self.hidden_size=hidden_size; self.embedding_dim=embedding_dim
        nfreq=n_fft//2+1; edges=torch.linspace(0,nfreq,num_bands+1).round().long().tolist()
        self.band_edges=[(edges[i],edges[i+1]) for i in range(num_bands)]
        if any(b<=a for a,b in self.band_edges): raise ValueError('num_bands too large for n_fft')
        widths=[b-a for a,b in self.band_edges]
        self.band_in=nn.ModuleList([nn.Linear(w,hidden_size) for w in widths])
        self.intra=nn.LSTM(hidden_size,hidden_size,batch_first=True,bidirectional=True)
        self.film=nn.Linear(embedding_dim,hidden_size*4)
        self.inter=nn.LSTM(hidden_size*2,hidden_size,batch_first=True,bidirectional=True)
        self.band_out=nn.ModuleList([nn.Linear(hidden_size*2,w) for w in widths])
        self.register_buffer('window',torch.hann_window(n_fft),persistent=False)
    def forward(self, mixture, speaker_embedding):
        if mixture.ndim!=2: raise ValueError('mixture must have shape [batch, samples]')
        if speaker_embedding.shape[-1]!=self.embedding_dim: raise ValueError(f'expected {self.embedding_dim}-D embedding')
        window=self.window.to(device=mixture.device,dtype=mixture.dtype)
        spec=torch.stft(mixture,n_fft=self.n_fft,hop_length=self.hop_length,win_length=self.n_fft,window=window,return_complex=True)
        mag=spec.abs(); features=[]
        for i,(start,end) in enumerate(self.band_edges):
            x=mag[:,start:end,:].transpose(1,2)
            x=self.band_in[i](x); x,_=self.intra(x); features.append(x)
        stacked=torch.stack(features,dim=2) # [B, frames, bands, 2H]
        gamma,beta=self.film(speaker_embedding).chunk(2,dim=-1)
        stacked=stacked*(1+gamma[:,None,None,:])+beta[:,None,None,:]
        b,t,bands,ch=stacked.shape
        x=stacked.reshape(b*t,bands,ch); x,_=self.inter(x); x=x.reshape(b,t,bands,ch)
        masks=[]
        for i in range(self.num_bands): masks.append(torch.sigmoid(self.band_out[i](x[:,:,i,:])).transpose(1,2))
        mask=torch.cat(masks,dim=1)
        enhanced_spec=spec*mask
        return torch.istft(enhanced_spec,n_fft=self.n_fft,hop_length=self.hop_length,win_length=self.n_fft,window=window,length=mixture.shape[-1])
