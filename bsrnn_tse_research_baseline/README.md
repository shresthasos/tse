# BSRNN-based Target Speaker Enhancement — Research Baseline

A reproducible baseline for target-speaker enhancement (TSE): synthetic mixture preparation, pretrained ECAPA-TDNN enrollment embeddings, a compact BSRNN-style model with speaker FiLM conditioning, training/resume, and research metric logging.

**Scope:** this is an independent baseline inspired by band-split recurrent networks, not a reproduction of PS4 or its proxy-supervised training. It uses a magnitude mask and mixture phase, a simple SI-SDR + waveform-L1 objective, and fixed ECAPA enrollment embeddings.

## Repository layout

```text
configs/default.json
scripts/prepare_data.py
scripts/precompute_embeddings.py
tse/{model,dataset,losses,utils}.py
train.py
evaluate.py
requirements*.txt
```

## Data layout

Use speaker-disjoint LibriSpeech folders (or compatible `speaker/chapter/audio` folders): `train-clean-100`, `dev-clean`, `test-clean`; optional DEMAND noise WAVs can be nested in a folder. Never point train/val/test at the same unpartitioned corpus. The prep script creates fixed-length mixtures, exact clean references, and separate same-speaker enrollment clips.

## Install

```bash
pip install -r requirements.txt
# Optional perceptual metrics:
pip install -r requirements-optional.txt
```

Use Python 3.10+ and a PyTorch/torchaudio pair compatible with your runtime. In Colab, select a GPU runtime first; don't reinstall torch unnecessarily if the preinstalled pair works.

## 1. Prepare data (start small)

```bash
python scripts/prepare_data.py \
 --train-dir /data/LibriSpeech/train-clean-100 \
 --val-dir /data/LibriSpeech/dev-clean \
 --test-dir /data/LibriSpeech/test-clean \
 --noise-dir /data/DEMAND --out data/processed \
 --samples-per-split 300 --duration 4 --seed 42
```

Omit `--noise-dir` for speech-on-speech mixtures only. Data is written as WAVs and JSONL manifests. Increase sample counts after the pilot works; generated audio consumes disk space.

## 2. Precompute speaker embeddings

```bash
python scripts/precompute_embeddings.py --data-root data/processed --device auto
```

The first run downloads SpeechBrain's `speechbrain/spkrec-ecapa-voxceleb` checkpoint. Embeddings are saved to `data/processed/embeddings/` and referenced by each manifest.

## 3. Train

```bash
python train.py --data-root data/processed --run-dir runs/bsrrn_baseline \
 --epochs 30 --batch-size 4 --lr 0.0003 --device auto
```

Use `--epochs 2 --batch-size 2` for a smoke test. Training resumes from `last.pt` unless `--no-resume` is supplied. Run outputs include `config.json`, `environment.json`, `history.csv`, `latest_metrics.json`, `best.pt`, and `last.pt`. Best model is selected by validation SI-SDRi.

## 4. Evaluate

```bash
python evaluate.py --manifest data/processed/test.jsonl \
 --checkpoint runs/bsrrn_baseline/best.pt --output-dir runs/bsrrn_baseline/test_eval --device auto
```

Optional ECAPA cosine similarities and STOI/PESQ:

```bash
python evaluate.py --manifest data/processed/test.jsonl \
 --checkpoint runs/bsrrn_baseline/best.pt --output-dir runs/bsrrn_baseline/test_eval \
 --device auto --speaker-metrics --perceptual-metrics
```

Outputs: `metrics_per_utterance.csv`, `metrics_summary.json`, and `enhanced/*.wav`. SI-SDRi = SI-SDR(enhanced, clean target) − SI-SDR(mixture, clean target). Positive SI-SDRi indicates improvement over the mixture. The clean target must be the exact mixture component and time-aligned; enrollment is not a test reference.

## Colab notes

Clone your GitHub repo, install requirements, mount Drive, and store `--run-dir` under `/content/drive/MyDrive/...` so checkpoints survive runtime resets. Keep data/manifests and model outputs backed up. Do not commit datasets, WAVs, or checkpoints to GitHub by default.

## Research reporting checklist

- Keep speaker identities disjoint across splits and never tune on test metrics.
- Record config, seed, package versions, manifest hashes, checkpoint, and per-utterance results.
- Compare models on exactly the same mixture recipes and reference WAVs.
- Report mean/median/std SI-SDRi and fraction improved; optionally STOI, PESQ, and target/enrollment embedding cosine similarity.
- State synthetic vs real recordings. Synthetic LibriSpeech results do not prove robustness on real meetings.
- Run multiple seeds if compute permits; don't compare metrics across incompatible datasets/protocols as directly equivalent.

## References

- PS4 repository (architecture and model-specific implementation): https://github.com/TaurenMountain/PS4
- PS4 model card: https://huggingface.co/TaurenMountain/PS4
- SpeechBrain ECAPA-TDNN: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- LibriSpeech: https://www.openslr.org/12
- DEMAND: https://zenodo.org/records/1227121

Choose a license before publishing and check the licenses/terms of all datasets and pretrained models separately.
