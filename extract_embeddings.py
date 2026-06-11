import os
import time
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.mixture import GaussianMixture
import shutil

import onnxruntime as ort
import torchaudio
import torchaudio.compliance.kaldi as kaldi


# configurations
MODEL_PATH = "ecapa_models/voxceleb_ECAPA512.onnx"
CLEAN_WAVS_DIR = 'VoiceBank+DEMAND/wav_clean'
NOISY_WAVS_DIR = 'VoiceBank+DEMAND/wav_noisy'
TRAINING_FILE_LIST = 'VoiceBank+DEMAND/training.txt'
SAVE_EMBEDDINGS_PATH = 'embeddings/voxceleb_ECAPA512'
NOISE_WAVS_TEMP_DIR = os.path.join(SAVE_EMBEDDINGS_PATH, 'noise_temp_wavs')
NUM_PROCESSES = 16
SAMPLE_RATE = 16000
# GMM configuration
NUM_GMM_COMPONENTS = 192 
GMM_COVARIANCE_TYPE = 'diag'
FEATURE_DIM = 192


class ONNXInferenceSession:
    _session = None
    _input_name = 'feats'
    _output_name = 'embs'

    @classmethod
    def get_session(cls, onnx_path):
        if cls._session is None:
            so = ort.SessionOptions()
            so.inter_op_num_threads = 1
            so.intra_op_num_threads = 1

            providers = ['CPUExecutionProvider']
            
            try:
                cls._session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
                print(f"ONNX model loaded successfully in process {os.getpid()} with provider: CPUExecutionProvider")
            except Exception as e:
                print(f"FATAL ERROR: Failed to load ONNX model on CPU in process {os.getpid()}: {e}")
                raise
        
        return cls._session, cls._input_name, cls._output_name

def compute_fbank(wav_path,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    waveform, sample_rate = torchaudio.load(wav_path)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)(waveform)
        sample_rate = SAMPLE_RATE
    
    waveform = waveform * (1 << 15)
    mat = kaldi.fbank(waveform,
                      num_mel_bins=num_mel_bins,
                      frame_length=frame_length,
                      frame_shift=frame_shift,
                      dither=dither,
                      sample_frequency=sample_rate,
                      window_type='hamming',
                      use_energy=False)
    mat = mat - torch.mean(mat, dim=0)
    return mat # [T, F]

def infer_onnx(wav_path):
    session, input_name, output_name = ONNXInferenceSession.get_session(MODEL_PATH)
    
    try:
        feats = compute_fbank(wav_path) # [T, F]
        feats = feats.unsqueeze(0).numpy()  # [1, T, F]
        
    except Exception as e:
        return None

    try:
        embeddings = session.run(output_names=[output_name],
                                 input_feed={input_name: feats})
        embedding = embeddings[0].squeeze(axis=0)
        
        if embedding.shape[0] != FEATURE_DIM:
            raise ValueError(f"Extracted feature dim {embedding.shape[0]} does not match expected {FEATURE_DIM}")

        return embedding
    except Exception as e:
        print(f"Error during ONNX inference for {wav_path}: {e}")
        return None

def compute_noise_wav(file_id, clean_path, noisy_path, noise_save_path):
    if os.path.exists(noise_save_path):
        return True
    
    try:
        clean_wav, sr_c = torchaudio.load(clean_path)
        noisy_wav, sr_n = torchaudio.load(noisy_path)
        if sr_c != SAMPLE_RATE or sr_n != SAMPLE_RATE:
            raise ValueError("Sample rates are not 16kHz.")
        L = min(clean_wav.shape[-1], noisy_wav.shape[-1])
        noise_wav = noisy_wav[..., :L] - clean_wav[..., :L]
        torchaudio.save(noise_save_path, noise_wav, SAMPLE_RATE)
        return True
    except Exception as e:
        print(f"Error computing/saving noise wav for {file_id}: {e}")
        return False

def save_embedding(embedding, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        tensor_to_save = torch.from_numpy(embedding).to(torch.float32).cpu()
        torch.save(tensor_to_save, save_path)
    except Exception as e:
        print(f"FATAL IO ERROR: Failed to save embedding to {save_path}. Error: {e}")
        raise

def process_file(file_id):
    clean_path = os.path.join(CLEAN_WAVS_DIR, f"{file_id}.wav")
    noisy_path = os.path.join(NOISY_WAVS_DIR, f"{file_id}.wav")
    noise_wav_path = os.path.join(NOISE_WAVS_TEMP_DIR, f"{file_id}.wav")

    save_clean_path = os.path.join(SAVE_EMBEDDINGS_PATH, "clean", f"{file_id}.pt")
    save_noisy_path = os.path.join(SAVE_EMBEDDINGS_PATH, "noisy", f"{file_id}.pt")
    save_noise_path = os.path.join(SAVE_EMBEDDINGS_PATH, "noise", f"{file_id}.pt")
    
    if os.path.exists(save_clean_path) and os.path.exists(save_noisy_path) and os.path.exists(save_noise_path):
        try:
            clean_embed = torch.load(save_clean_path).numpy()
            noisy_embed = torch.load(save_noisy_path).numpy()
            noise_embed = torch.load(save_noise_path).numpy()
            return clean_embed, noisy_embed, noise_embed
        except Exception:
            pass

    if not os.path.exists(clean_path) or not os.path.exists(noisy_path):
        return None

    if not compute_noise_wav(file_id, clean_path, noisy_path, noise_wav_path):
        return None

    clean_embed = infer_onnx(clean_path)
    noisy_embed = infer_onnx(noisy_path)
    noise_embed = infer_onnx(noise_wav_path) 

    if clean_embed is None or noisy_embed is None or noise_embed is None:
        return None

    save_embedding(clean_embed, save_clean_path)
    save_embedding(noisy_embed, save_noisy_path)
    save_embedding(noise_embed, save_noise_path)
    
    return clean_embed, noisy_embed, noise_embed

def fit_and_save_gmm_prior_full(embeddings_list, feature_name):
    if not embeddings_list:
        print(f"No valid {feature_name} embeddings for GMM fitting.")
        return None
    
    X = np.stack(embeddings_list)
    K_fit = min(NUM_GMM_COMPONENTS, X.shape[0])
    
    if K_fit < NUM_GMM_COMPONENTS:
        print(f"[WARNING] Reducing GMM components for {feature_name} from {NUM_GMM_COMPONENTS} to {K_fit} due to data size.")

    print(f"Fitting GMM for {feature_name} data shape: {X.shape} with K={K_fit}")

    gmm = GaussianMixture(
        n_components=K_fit,
        covariance_type=GMM_COVARIANCE_TYPE,
        random_state=0,
        max_iter=500
    )
    gmm.fit(X)

    mu_k = gmm.means_
    sigma2_k = gmm.covariances_
    weights_k = gmm.weights_
    
    save_embedding(mu_k, os.path.join(SAVE_EMBEDDINGS_PATH, f"{feature_name}_gmm_mu_k.pt"))
    save_embedding(sigma2_k, os.path.join(SAVE_EMBEDDINGS_PATH, f"{feature_name}_gmm_sigma2_k.pt"))
    save_embedding(weights_k, os.path.join(SAVE_EMBEDDINGS_PATH, f"{feature_name}_gmm_weights_k.pt"))

    mean_embed = np.mean(X, axis=0)
    save_embedding(mean_embed, os.path.join(SAVE_EMBEDDINGS_PATH, f"{feature_name}_mean.pt"))

    print(f"GMM Full Prior saved: {feature_name} (K={K_fit}, D={mu_k.shape[1]})")
    
    return mu_k, sigma2_k, weights_k

def compute_and_save_embeddings():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"ONNX model not found at {MODEL_PATH}")

    os.makedirs(NOISE_WAVS_TEMP_DIR, exist_ok=True)

    file_list = []
    with open(TRAINING_FILE_LIST, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                file_id = line.split('|')[0]
                file_list.append(file_id)

    print(f"Total files found: {len(file_list)}. Starting extraction with {NUM_PROCESSES} processes.")
    print(f"Expected Feature Dimension (D_E): {FEATURE_DIM}")

    clean_embeddings, noisy_embeddings, noise_embeddings = [], [], []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        futures = {executor.submit(process_file, file_id): file_id for file_id in file_list}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            try:
                result = future.result()
                if result is not None:
                    clean_embed, noisy_embed, noise_embed = result
                    clean_embeddings.append(clean_embed)
                    noisy_embeddings.append(noisy_embed)
                    noise_embeddings.append(noise_embed)
            except Exception as e:
                print(f"\nFATAL ERROR in worker process: {e}")

    try:
        shutil.rmtree(NOISE_WAVS_TEMP_DIR)
        print(f"Cleaned up temporary noise wavs in {NOISE_WAVS_TEMP_DIR}")
    except OSError as e:
        print(f"Error removing temporary directory {NOISE_WAVS_TEMP_DIR}: {e}")

    if not clean_embeddings or not noisy_embeddings or not noise_embeddings:
        print("No valid embeddings were extracted. Check logs for ONNX inference errors.")
        return None

    print("\n--- Starting GMM Fitting ---")
    
    fit_and_save_gmm_prior_full(clean_embeddings, "clean")
    fit_and_save_gmm_prior_full(noise_embeddings, "noise")
    fit_and_save_gmm_prior_full(noisy_embeddings, "noisy")

    print(f"\n--- Extraction & GMM Fitting Finished ---")
    print(f"Total time: {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
    
    compute_and_save_embeddings()