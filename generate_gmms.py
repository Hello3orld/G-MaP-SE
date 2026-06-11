import os
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.mixture import GaussianMixture


def save_distribution_param(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tensor_to_save = torch.from_numpy(data).to(torch.float32).cpu()
    torch.save(tensor_to_save, save_path)

def load_single_pt(file_path):
    try:
        return torch.load(file_path, map_location='cpu').numpy().astype(np.float32)
    except Exception:
        return None

def load_all_embeddings(dataset_paths, max_workers=24):
    all_file_paths = []
    
    for folder_path in tqdm(dataset_paths, desc="Scanning Directories"):
        if os.path.exists(folder_path):
            for root, _, files in os.walk(folder_path):
                all_file_paths.extend([os.path.join(root, f) for f in files if f.endswith('.pt')])
        else:
            print(f"\n[Warning] Path not found: {folder_path}")

    num_files = len(all_file_paths)
    if num_files == 0:
        raise ValueError("No .pt files found!")
    print(f"Total files found: {num_files}")

    all_embeddings = []
    
    chunk_size = 1000 
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        pbar = tqdm(total=num_files, desc="Loading Embeddings", unit="file")
        for i in range(0, num_files, chunk_size):
            chunk = all_file_paths[i : i + chunk_size]
            futures = [executor.submit(load_single_pt, fp) for fp in chunk]
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    all_embeddings.append(res)
                pbar.update(1)
        pbar.close()

    X = np.vstack(all_embeddings)
    del all_embeddings
    print(f"Data ready. Shape: {X.shape}, Memory usage: ~{X.nbytes / 1024**3:.2f} GB")
    return X

def normalize_data(X):
    norms = np.linalg.norm(X, ord=2, axis=1, keepdims=True)
    return X / (norms + 1e-8)

def fit_gmm_for_ks(X, k_list, base_output_dir):
    global_mean = np.mean(X, axis=0)

    for k in k_list:
        output_dir = base_output_dir.replace("{K}", str(k))
        
        k_fit = min(k, X.shape[0])
        
        gmm = GaussianMixture(
            n_components=k_fit,
            covariance_type='diag',
            random_state=42,
            max_iter=50000,
            verbose=1 
        )
        
        gmm.fit(X)

        save_distribution_param(gmm.means_, os.path.join(output_dir, "clean_gmm_mu_k.pt"))
        save_distribution_param(gmm.covariances_, os.path.join(output_dir, "clean_gmm_sigma2_k.pt"))
        save_distribution_param(gmm.weights_, os.path.join(output_dir, "clean_gmm_weights_k.pt"))
        save_distribution_param(global_mean, os.path.join(output_dir, "clean_mean.pt"))
        
        print(f"Saved results for K={k}")

if __name__ == "__main__":
    NUM_WORKERS = 24
    K_LIST = [192]
    VBD_path = 'embeddings/voxceleb_ECAPA512/clean'
    DATASET_LIST = [VBD_path]
    BASE_OUTPUT_DIR = "gmms"

    try:
        X_data = load_all_embeddings(DATASET_LIST, max_workers=NUM_WORKERS)
        X_norm = normalize_data(X_data)
        fit_gmm_for_ks(X_norm, K_LIST, BASE_OUTPUT_DIR)
        print("\n--- All Done ---")
    except Exception as e:
        import traceback
        traceback.print_exc()
