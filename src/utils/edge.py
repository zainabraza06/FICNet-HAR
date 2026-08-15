import time
import pickle
import numpy as np
import torch
from src.config import DEVICE

def measure_inference_time_classical(model, X_test, n_runs=500):
    for _ in range(50): _ = model.predict(X_test[:100])
    times = []
    for _ in range(n_runs):
        start = time.perf_counter(); _ = model.predict(X_test[:100]); end = time.perf_counter()
        times.append((end-start)*1000)
    return np.mean(times), np.std(times)

def measure_inference_time_bilstm(model, Xb_test, n_runs=500):
    model.eval()
    Xb_t = torch.tensor(Xb_test[:100], dtype=torch.float32).to(DEVICE)
    for _ in range(50):
        with torch.no_grad(): _ = model(Xb_t)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad(): _ = model(Xb_t)
        end = time.perf_counter(); times.append((end-start)*1000)
    return np.mean(times), np.std(times)

def measure_inference_time_fusion(model, Xb_test, Xf_test, n_runs=500):
    model.eval()
    Xb_t = torch.tensor(Xb_test[:100], dtype=torch.float32).to(DEVICE)
    Xf_t = torch.tensor(Xf_test[:100], dtype=torch.float32).to(DEVICE)
    for _ in range(50):
        with torch.no_grad(): _ = model(Xb_t, Xf_t)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad(): _ = model(Xb_t, Xf_t)
        end = time.perf_counter(); times.append((end-start)*1000)
    return np.mean(times), np.std(times)

def measure_model_size(model, is_torch=False):
    if is_torch:
        param_count = sum(p.numel() for p in model.parameters())
        size_bytes = param_count * 4 * 2
    else:
        size_bytes = len(pickle.dumps(model))
    return size_bytes / (1024 * 1024)
