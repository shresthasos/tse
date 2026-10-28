# Google Colab quick start

1. Select Runtime → Change runtime type → GPU.
2. Clone your GitHub repo and install dependencies:

```python
!git clone https://github.com/<USERNAME>/<REPO>.git
%cd <REPO>
!pip install -r requirements.txt
```

3. Download/extract LibriSpeech train-clean-100, dev-clean, test-clean and optionally DEMAND under `/content/data/`.
4. Mount Drive before training to persist runs:

```python
from google.colab import drive
drive.mount('/content/drive')
!python scripts/prepare_data.py --train-dir /content/data/LibriSpeech/train-clean-100 --val-dir /content/data/LibriSpeech/dev-clean --test-dir /content/data/LibriSpeech/test-clean --noise-dir /content/data/DEMAND --out /content/data/processed --samples-per-split 200 --duration 4 --seed 42
!python scripts/precompute_embeddings.py --data-root /content/data/processed --device auto
!python train.py --data-root /content/data/processed --run-dir /content/drive/MyDrive/bsrrn_tse/runs/pilot --epochs 2 --batch-size 2 --device auto
!python evaluate.py --manifest /content/data/processed/test.jsonl --checkpoint /content/drive/MyDrive/bsrrn_tse/runs/pilot/best.pt --output-dir /content/drive/MyDrive/bsrrn_tse/runs/pilot/test_eval --device auto
```

Use smaller batch sizes if CUDA runs out of memory. Generated mixtures and embeddings consume storage. `/content` is ephemeral; Drive-backed run directories persist across runtime resets.
