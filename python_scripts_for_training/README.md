# Keyword Spotting (KWS) with Energy-Efficient Deep Learning

A framework for training, fine-tuning, and deploying lightweight CNN models for keyword spotting on edge devices. This project focuses on energy efficiency, model compression, and deployment on resource-constrained hardware.

## Features

- **10 CNN architectures** ranging from lightweight (8 filters) to deep (512 filters)
- **10-fold cross-validation** with speaker-independent splits
- **Fine-tuning pipeline** for domain adaptation
- **Energy monitoring** using Intel RAPL (CPU) and NVML (GPU)
- **TFLite conversion** with INT8 quantization
- **Comprehensive visualization** of accuracy, energy consumption, and model efficiency
- **Microcontroller deployment** with C array conversion

---

## System Requirements

### Hardware
- **CPU**: Intel processor with RAPL support (for CPU energy monitoring)
- **GPU**: NVIDIA GPU with compute capability ≥3.5 (optional, for GPU acceleration)
- **RAM**: Minimum 16GB recommended
- **Storage**: ~10GB for dataset and models

### Software
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.10.x
- **TensorFlow**: 2.11.1 (compatible with TFLite and CUDA 11.2)
- **CUDA**: 11.2
- **cuDNN**: 8.1.0

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/kws-energy-efficient.git
cd kws-energy-efficient
```

### 2. Create Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install TensorFlow 2.11.1 (GPU version)
pip install tensorflow==2.11.1

# Install other dependencies
pip install numpy==1.23.5
pip install librosa==0.10.0
pip install python_speech_features==0.6
pip install scikit-learn==1.3.0
pip install matplotlib==3.7.1
pip install pandas==2.0.3
pip install psutil==5.9.5
pip install pynvml==11.5.0
```

### 4. Install CUDA and cuDNN

**For Ubuntu:**

```bash
# CUDA 11.2
wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.27.04_linux.run
sudo sh cuda_11.2.0_460.27.04_linux.run

# cuDNN 8.1.0
# Download from NVIDIA website (requires account)
# https://developer.nvidia.com/rdp/cudnn-archive
tar -xzvf cudnn-11.2-linux-x64-v8.1.0.77.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

**Add to ~/.bashrc:**

```bash
export PATH=/usr/local/cuda-11.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH
```

### 5. Enable RAPL for CPU Energy Monitoring

```bash
# Grant read permissions to RAPL interface
sudo chmod -R a+r /sys/class/powercap/intel-rapl/
```

### 6. Verify Installation

```bash
python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"
```

Expected output:
```
TF version: 2.11.1
GPU available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## Dataset Preparation

This project uses the **Google Speech Commands Dataset v0.02**.

### Download Dataset

```bash
# Create dataset directory
mkdir -p data
cd data

# Download and extract
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir speech_commands_v0.02
tar -xzf speech_commands_v0.02.tar.gz -C speech_commands_v0.02
cd ..
```

### Dataset Structure

```
speech_commands_v0.02/
├── yes/
│   ├── 0a7c2a8d_nohash_0.wav
│   └── ...
├── no/
├── up/
├── down/
├── _background_noise_/
└── ... (36 classes total)
```

### Create K-Fold Splits

Run the ETL notebook to generate 10-fold cross-validation splits:

```bash
jupyter notebook KWS_ETL.ipynb
```

Update the `dataset_path` variable in the notebook:
```python
dataset_path = "data/speech_commands_v0.02"
```

This will create `new_Data_particions/fold_1.npz` through `fold_10.npz` containing:
- Training set (81%)
- Validation set (9%)
- Test set (10%)
- Fine-tuning set (30% of test set)
- Final test set (70% of test set)

---

## Usage Guide

### 1. Train Models

Open `KWS_model_training.ipynb` and train individual models:

```python
# Train a single model (e.g., Model 1)
metrics_1 = train_model_with_metrics(
    build_model_1, 
    'model_1', 
    early_stopping_patience=5, 
    monitor_resources=True
)
```

**Training parameters:**
- `epochs=30` (with early stopping)
- `batch_size=128`
- `early_stopping_patience=5`

**Output:**
- Trained models saved to `trained_models/model_X_fold_Y_best.h5`
- Metrics saved to `saved_metrics/model_X_metrics.pkl`

### 2. Fine-tune Models

Fine-tune trained models on the fine-tuning dataset:

```python
# Fine-tune a single model
finetuned_metrics_1 = finetune_all_folds(
    model_name='model_1',
    epochs=10,
    batch_size=128,
    monitor_resources=True
)
```

**Output:**
- Fine-tuned models saved to `finetuned_models/model_X_fold_Y_finetuned.h5`
- Metrics saved to `saved_finetuned_metrics/`

### 3. Measure Test Energy

Measure energy consumption during inference:

```python
# Measure for normal models
df_normal = measure_all_test_power(
    models_dir='trained_models',
    output_file='saved_metrics/normal_models_test_power.csv'
)

# Measure for fine-tuned models
df_finetuned = measure_all_test_power(
    models_dir='finetuned_models',
    output_file='saved_finetuned_metrics/finetuned_models_test_power.csv',
    is_finetuned=True
)
```

### 4. Convert to TFLite

Open `KWS_tflite_conversion.ipynb`:

```python
# Convert normal models
convert_all_models(
    models_dir='trained_models',
    folds_dir='new_Data_particions',
    output_base_dir='tflite_models/normal',
    is_finetuned=False
)

# Convert fine-tuned models
convert_all_models(
    models_dir='finetuned_models',
    folds_dir='new_Data_particions',
    output_base_dir='tflite_models/finetuned',
    is_finetuned=True
)
```

**Output formats:**
- `no_quantization/` - Full precision TFLite models
- `int8_quantization/` - INT8 quantized models (4x smaller)

### 5. Evaluate TFLite Models

```python
# Evaluate on CPU
df_cpu = evaluate_all_tflite_models(
    tflite_base_dir='tflite_models',
    folds_dir='new_Data_particions',
    output_csv='converted_metrics/tflite_normal_no_quant_cpu_results.csv',
    model_type='normal',
    quantization='no_quant',
    use_gpu=False
)

# Evaluate on GPU (if available)
df_gpu = evaluate_all_tflite_models(
    output_csv='converted_metrics/tflite_normal_no_quant_gpu_results.csv',
    use_gpu=True
)

# Evaluate INT8 models
df_int8 = evaluate_all_tflite_models(
    output_csv='converted_metrics/tflite_normal_int8_results.csv',
    quantization='int8'
)
```

### 6. Generate Visualizations

Open `KWS_visualizations.ipynb` to generate comparison plots:

```python
# Load all metrics
with open('saved_metrics/all_metrics.pkl', 'rb') as f:
    all_metrics = pickle.load(f)

# Generate energy comparison plots
# All plots saved to figures/energy_comparisons/
```

**Generated visualizations:**
1. Normal vs Fine-tuned energy consumption
2. Keras vs TFLite (no quantization)
3. Keras vs TFLite (INT8)
4. TFLite no-quant vs INT8
5. Accuracy distributions
6. F1-score comparisons

---

## Model Architectures

| Model | Conv Layers | Filters | Dense Units | Dropout | Parameters |
|-------|-------------|---------|-------------|---------|------------|
| Model 1 | 1 | 8 | 32 | No | ~2K |
| Model 2 | 1 | 16 | 64 | No | ~6K |
| Model 3 | 2 | 16→32 | 128 | Yes | ~18K |
| Model 4 | 1 | 32 | 128 | No | ~20K |
| Model 5 | 2 | 16→32 | 64 | No | ~11K |
| Model 6 | 2 | 16→32 | 128 | Yes | ~18K |
| Model 7 | 3 | 32→64→128 | 128 | Yes | ~95K |
| Model 8 | 2 | 64→128 | 256 | Yes | ~210K |
| Model 9 | 3 | 64→128→256 | 512 | Yes | ~580K |
| Model 10 | 3 | 128→256→512 | 1024 | Yes | ~2.3M |

**Input:** 13 MFCC coefficients × 50 frames (1 second of audio at 8kHz)

**Output:** 36 classes (keywords + background noise)

---

## Energy Monitoring

### CPU Energy (Intel RAPL)

Monitors package and DRAM energy consumption via `/sys/class/powercap/intel-rapl/`.

**Requirements:**
- Intel CPU with RAPL support (Sandy Bridge or newer)
- Read permissions: `sudo chmod -R a+r /sys/class/powercap/intel-rapl/`

### GPU Energy (NVIDIA NVML)

Monitors GPU power draw using NVIDIA Management Library.

**Requirements:**
- NVIDIA GPU with power monitoring support
- `pynvml` library installed

### Metrics Collected

- **CPU/GPU utilization** (%)
- **Power consumption** (Watts)
- **Energy consumption** (Watt-hours)
- **Memory usage** (MB)
- **Training/inference duration** (seconds)

---

## TFLite Deployment

### Compatibility

TensorFlow 2.11.1 is **compatible** with TFLite for both:
- Full precision models (float32)
- INT8 quantized models

### Quantization Benefits

| Metric | Float32 | INT8 | Improvement |
|--------|---------|------|-------------|
| Model Size | ~100 KB | ~25 KB | **4x smaller** |
| Inference Speed | 1.0x | 2-4x | **2-4x faster** |
| Accuracy | 95.2% | 94.8% | -0.4% |
| Energy | 1.0x | 0.6x | **40% less** |

### Device Support

- **CPU**: All quantization types supported
- **GPU**: Only float32 supported (GPU delegate)
- **Edge TPU**: INT8 models with full integer quantization
- **Microcontrollers**: INT8 models via TensorFlow Lite Micro

---

## Microcontroller Deployment

### Convert TFLite to C Array

Use the `xxd` tool or Python script to convert `.tflite` models to C arrays:

#### Using xxd (Linux/Mac)

```bash
# Navigate to your TFLite model directory
cd tflite_models/normal/int8_quantization

# Convert to C array
xxd -i model_1_fold_1.tflite > model_1_fold_1.cc

# Or create header file
echo "alignas(8) const unsigned char model_data[] = {" > model_array.h
xxd -i < model_1_fold_1.tflite | sed 's/unsigned/const unsigned/g' >> model_array.h
echo "};" >> model_array.h
echo "const int model_data_len = sizeof(model_data);" >> model_array.h
```


## Troubleshooting

### TensorFlow GPU Not Detected

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Verify TensorFlow GPU build
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### RAPL Permission Denied

```bash
# Grant read permissions
sudo chmod -R a+r /sys/class/powercap/intel-rapl/

# Make persistent (add to /etc/rc.local)
```

### TFLite GPU Delegate Error

INT8 quantized models don't support GPU acceleration. Use CPU for INT8 or convert to float16 for GPU.

### Memory Issues During Training

Reduce batch size or use memory cleanup:
```python
tf.keras.backend.clear_session()
gc.collect()
```
