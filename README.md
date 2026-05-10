# 🔎 AUDITIA ROCm-X: Public Contract Auditor

**AUDITIA** is a high-performance, open-source AI auditing agent engineered to combat opacity in Colombian public contracting. Built natively on the AMD Instinct MI300X, it leverages Google's JAX and Flax frameworks to run an unsupervised Autoencoder capable of analyzing millions of public records at extreme speeds.

This project was built for the **AMD Developer Cloud Hackathon (Track 2: Fine-Tuning on AMD GPUs)**.

## 🧠 Architecture & Technical Walkthrough

AUDITIA moves away from traditional rule-based auditing by mathematically learning the "normality" of public spending and flagging outliers.

### 1. The Data Pipeline (Polars & HuggingFace)
We process real Colombian public contracts from the SECOP II API. The data is vectorized into a precise 460-dimensional tensor:
* **Numerical Data (Value & Time):** Normalized using Z-scores to prevent massive budgets from mathematically crushing execution timeframes.
* **Categorical Data:** Processed via One-Hot Encoding for structural data (departments, contract types).
* **Free-Text (Contract Object):** Processed locally using the `paraphrase-multilingual-MiniLM-L12-v2` transformer to generate dense semantic embeddings (384 dimensions).

### 2. The Core AI Engine (JAX + Flax)
* **Unsupervised Autoencoder:** We built a Multi-Layer Perceptron using `flax.linen`. The model is trained on 5.6 million historical contracts to compress and reconstruct the data.
* **Anomaly Score:** When an irregular contract (e.g., absurdly inflated price, extremely vague description) is processed, the model fails to reconstruct it. The resulting Mean Squared Error (MSE) serves as an objective, mathematical Anomaly Score.
* **Extreme Acceleration:** By using `@jax.jit`, the JAX XLA compiler fuses the matrix operations and translates them into native instructions for the AMD MI300X.

### 3. Production Deployment (FastAPI)
The inference engine is wrapped in an asynchronous FastAPI server. It performs a "hot load" of the `.msgpack` (model weights) and `.joblib` (scalers/encoders) files directly into the MI300X's VRAM, allowing for millisecond-latency predictions.

## 🛠️ Building on AMD MI300X: Developer Insights

Building this architecture on the AMD Developer Cloud and ROCm 6.4.2 presented unique low-level engineering challenges that we successfully solved:

* **JAX vs. MIOpen VRAM Preallocation:** JAX naturally attempts to preallocate 90% of the VRAM. On the MI300X, this starved AMD's MIOpen (the underlying DNN library), causing `miopenStatusUnknownError`. We solved this by strictly enforcing `XLA_PYTHON_CLIENT_PREALLOCATE=false`, allowing dynamic memory sharing.
* **Driver Hijacking:** Mixing PyTorch-based libraries (`sentence-transformers`) with JAX caused context collisions (`HIP_ERROR_NoDevice`). We mitigated this by strictly loading JAX and MIOpen first, and forcing the text embedding model to `device='cpu'`, reserving the massive MI300X compute entirely for the XLA Autoencoder.
* **Docker Cache Permissions:** To prevent MIOpen from crashing with "No device" due to root cache permission blocks inside our container, we redirected the kernel compilation cache to `/tmp` using `MIOPEN_USER_DB_PATH` and `MIOPEN_CUSTOM_CACHE_DIR`.

## 🚀 How to Run (AMD ROCm Environment)

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/auditia-rocm-x.git](https://github.com/your-username/auditia-rocm-x.git)
cd auditia-rocm-x
```
**2. Install strict dependencies (Avoiding CUDA binaries):**
pip install jax==0.4.35
pip install --no-deps flax==0.8.5 optax==0.2.3
pip install polars fastapi uvicorn scikit-learn sentence-transformers
**3. Set ROCm Environment Variables:**
export ROCM_PATH=/opt/rocm
export MIOPEN_DEBUG_INSTALL_LOCATION=$ROCM_PATH
export MIOPEN_DISABLE_CACHE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
**4. Start the Inference API:**
python main.py
