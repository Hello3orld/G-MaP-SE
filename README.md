# G-MaP-SE

Official PyTorch implementation of [G-MaP-SE](https://arxiv.org/pdf/2606.08580v1)

## Prerequisites

### Environment Setup

```bash
pip install -r requirements.txt
```

### Data Preparation

Download and extract the [VoiceBank+DEMAND dataset](https://datashare.ed.ac.uk/handle/10283/1942). Resample all wav files to 16kHz, and place the audio files as follows:

```
VoiceBank+DEMAND/
├── training.txt          # Training file list
├── test.txt              # Test file list
├── wav_clean/            # Clean speech waveforms
└── wav_noisy/            # Noisy speech waveforms
```

### Pre-trained Speaker Encoder

Download the [ECAPA-TDNN speaker encoder model](https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_ECAPA512.onnx) and place it as `ecapa_models/voxceleb_ECAPA512.onnx`.


## Extract Speaker Embeddings

Extract ECAPA-TDNN speaker embeddings for the training dataset:

```bash
python extract_embeddings.py
```

This will generate embeddings saved in `embeddings/voxceleb_ECAPA512/{clean,noisy,noise}/`.

## Train GMM Priors

Fit Gaussian Mixture Models (GMMs) on the extracted speaker embeddings:

```bash
python generate_gmms.py
```

This will generate GMM parameters saved in `gmms/`.

## Training

Train the G-MaP-SE model:

```bash
python train_g_map_se.py --config config_g_map_se.json --checkpoint_path ckpt/g_map_se
```

## Inference

Run inference on noisy speech:

```bash
python inference_g_map_se.py --input_noisy_wavs_dir <path> --output_dir <output_path> --checkpoint_file ckpt/g_best
```

## Acknowledgements

We referred to [MP-SENet](https://github.com/jik876/hifi-gan) to implement this.