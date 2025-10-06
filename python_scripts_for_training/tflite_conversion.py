import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model_training import precision_m, recall_m, f1_m
from resource_monitoring import ResourceMonitor
import pandas as pd
import time

def load_representative_data(fold_path, num_samples=1000):
    """
    Load representative training data for quantization calibration.
    
    Args:
        fold_path: Path to fold .npz file
        num_samples: Number of samples to use for calibration
    
    Returns:
        Representative dataset as numpy array
    """
    data = np.load(fold_path, allow_pickle=True)
    x_train = np.array(data['x_train_fold'])
    
    # Use subset for calibration
    if len(x_train) > num_samples:
        indices = np.random.choice(len(x_train), num_samples, replace=False)
        x_train = x_train[indices]
    
    return x_train.astype(np.float32)


def convert_to_tflite_no_quant(model_path, tflite_path):
    """
    Convert Keras model to TFLite without quantization.
    
    Args:
        model_path: Path to saved Keras model (.h5)
        tflite_path: Path to save TFLite model
    """

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'precision_m': precision_m,
            'recall_m': recall_m,
            'f1_m': f1_m
        }
    )
    
    # Convert without quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"  Saved (no quant): {tflite_path} ({size_kb:.1f} KB)")


def convert_to_tflite_int8(model_path, tflite_path, representative_data):
    """
    Convert Keras model to TFLite with int8 quantization.
    
    Args:
        model_path: Path to saved Keras model (.h5)
        tflite_path: Path to save TFLite model
        representative_data: Numpy array for quantization calibration
    """
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'precision_m': precision_m,
            'recall_m': recall_m,
            'f1_m': f1_m
        }
    )
    
    def representative_data_gen():
        for i in range(len(representative_data)):
            yield [representative_data[i:i+1]]
    
    # Convert with int8 quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"  Saved (int8):    {tflite_path} ({size_kb:.1f} KB)")


def convert_all_models(models_dir, folds_dir, output_base_dir, 
                       num_models=10, num_folds=10, is_finetuned=False):
    """
    Convert all models to TFLite in both formats.
    
    Args:
        models_dir: Directory with trained models
        folds_dir: Directory with fold data (for representative data)
        output_base_dir: Base directory for TFLite models
        num_models: Number of models
        num_folds: Number of folds
        is_finetuned: Whether these are fine-tuned models
    """
    no_quant_dir = os.path.join(output_base_dir, 'no_quantization')
    int8_dir = os.path.join(output_base_dir, 'int8_quantization')
    os.makedirs(no_quant_dir, exist_ok=True)
    os.makedirs(int8_dir, exist_ok=True)
    
    model_type = "Fine-tuned" if is_finetuned else "Normal"
    print(f"\n{'='*80}")
    print(f"CONVERTING {model_type.upper()} MODELS TO TFLITE")
    print(f"{'='*80}\n")
    
    total_converted = 0
    
    for model_idx in range(1, num_models + 1):
        model_name = f'model_{model_idx}'
        print(f"\n{model_name}:")
        
        for fold_no in range(1, num_folds + 1):
            if is_finetuned:
                model_path = os.path.join(models_dir, f'{model_name}_fold_{fold_no}_finetuned.h5')
                base_name = f'{model_name}_fold_{fold_no}_finetuned'
            else:
                model_path = os.path.join(models_dir, f'{model_name}_fold_{fold_no}_best.h5')
                base_name = f'{model_name}_fold_{fold_no}'
            
            # Check if model exists
            if not os.path.exists(model_path):
                print(f"  Fold {fold_no}: Model not found, skipping")
                continue
            
            print(f"  Fold {fold_no}:")
            
            try:
                fold_path = os.path.join(folds_dir, f'fold_{fold_no}.npz')
                rep_data = load_representative_data(fold_path, num_samples=1000)
                
                # Convert without quantization
                tflite_no_quant = os.path.join(no_quant_dir, f'{base_name}.tflite')
                convert_to_tflite_no_quant(model_path, tflite_no_quant)
                
                # Convert with int8 quantization
                tflite_int8 = os.path.join(int8_dir, f'{base_name}.tflite')
                convert_to_tflite_int8(model_path, tflite_int8, rep_data)
                
                total_converted += 1
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
            
            tf.keras.backend.clear_session()
    
    print(f"\n{'='*80}")
    print(f"CONVERSION COMPLETE: {total_converted}/{num_models * num_folds} models converted")
    print(f"{'='*80}\n")
    
    return total_converted

def load_test_data(fold_path, num_classes=36):
    """Load test data from fold file."""
    data = np.load(fold_path, allow_pickle=True)
    x_test = np.array(data['x_final_test'])
    y_test = np.array(data['y_final_test'])
    return x_test, y_test


def evaluate_tflite_model(tflite_path, x_test, y_test, is_quantized=False, use_gpu=False):
    """
    Evaluate a TFLite model with optional GPU acceleration.
    
    Args:
        tflite_path: Path to TFLite model
        x_test: Test features
        y_test: Test labels
        is_quantized: Whether model is int8 quantized
        use_gpu: Whether to attempt GPU acceleration
    
    Returns:
        Dictionary with metrics and resource stats
    """
    delegate_info = "CPU"
    
    if use_gpu and not is_quantized:
        try:
            try:
                # Method 1: Load from system library
                gpu_delegate = tf.lite.experimental.load_delegate('libgpu_delegate.so')
                interpreter = tf.lite.Interpreter(
                    model_path=tflite_path,
                    experimental_delegates=[gpu_delegate]
                )
                delegate_info = "GPU (libgpu_delegate.so)"
            except:
                # Method 2: Use built-in GPU delegate
                interpreter = tf.lite.Interpreter(
                    model_path=tflite_path,
                    experimental_delegates=[tf.lite.experimental.load_delegate('gpu')]
                )
                delegate_info = "GPU (built-in)"
        except Exception as e:
            # Fallback to CPU
            print(f"    GPU delegate failed: {e}")
            print(f"    Falling back to CPU")
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            delegate_info = "CPU (GPU failed)"
    elif is_quantized and use_gpu:
        # Int8 quantization not supported on GPU
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        delegate_info = "CPU (int8 not GPU-compatible)"
    else:
        # Use CPU
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        delegate_info = "CPU"
    
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_scale = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]
    output_scale = output_details[0]['quantization'][0]
    output_zero_point = output_details[0]['quantization'][1]

    for _ in range(5):
        sample = x_test[:1].astype(np.float32)
        if is_quantized:
            sample = (sample / input_scale + input_zero_point).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()

    monitor = ResourceMonitor(log_interval=0.1)
    monitor.start()

    predictions = []
    for i in range(len(x_test)):
        input_data = x_test[i:i+1].astype(np.float32)

        if is_quantized:
            input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])

        if is_quantized:
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        predictions.append(np.argmax(output_data))

    resource_stats = monitor.stop()

    y_pred = np.array(predictions)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'duration_s': resource_stats['duration_seconds'],
        'gpu_power_mean_w': resource_stats.get('gpu_power_mean_w', 0),
        'gpu_power_max_w': resource_stats.get('gpu_power_max_w', 0),
        'gpu_usage_mean': resource_stats.get('gpu_usage_mean', 0),
        'gpu_energy_wh': resource_stats.get('gpu_energy_wh', 0),
        'cpu_power_mean_w': resource_stats.get('cpu_power_mean_w', 0),  # Add this
        'cpu_power_max_w': resource_stats.get('cpu_power_max_w', 0),    # Add this
        'cpu_energy_wh': resource_stats.get('cpu_energy_wh', 0),        # Add this
        'energy_wh': resource_stats.get('cpu_energy_wh', 0),  # This should be CPU for TFLite
        'model_size_kb': os.path.getsize(tflite_path) / 1024,
        'delegate': delegate_info
    }


def evaluate_all_tflite_models(tflite_base_dir, folds_dir, output_csv,
                                num_models=10, num_folds=10, 
                                model_type='normal', quantization='no_quant',
                                use_gpu=False):
    """
    Evaluate all TFLite models of a specific type.
    
    Args:
        tflite_base_dir: Base directory for TFLite models
        folds_dir: Directory with fold data
        output_csv: CSV file to save results
        num_models: Number of models
        num_folds: Number of folds
        model_type: 'normal' or 'finetuned'
        quantization: 'no_quant' or 'int8'
        use_gpu: Whether to attempt GPU acceleration
    """
    if quantization == 'int8':
        models_dir = os.path.join(tflite_base_dir, model_type, 'int8_quantization')
        is_quantized = True
    else:
        models_dir = os.path.join(tflite_base_dir, model_type, 'no_quantization')
        is_quantized = False
    
    device_type = "GPU" if (use_gpu and not is_quantized) else "CPU"
    
    print(f"\n{'='*80}")
    print(f"EVALUATING {model_type.upper()} TFLITE MODELS ({quantization.upper()}) on {device_type}")
    print(f"{'='*80}\n")
    
    results = []
    
    for model_idx in range(1, num_models + 1):
        model_name = f'model_{model_idx}'
        print(f"\n{model_name}:")
        
        for fold_no in range(1, num_folds + 1):
            if model_type == 'finetuned':
                tflite_path = os.path.join(models_dir, f'{model_name}_fold_{fold_no}_finetuned.tflite')
            else:
                tflite_path = os.path.join(models_dir, f'{model_name}_fold_{fold_no}.tflite')
            
            if not os.path.exists(tflite_path):
                print(f"  Fold {fold_no}: Model not found, skipping")
                continue
            
            fold_path = os.path.join(folds_dir, f'fold_{fold_no}.npz')
            x_test, y_test = load_test_data(fold_path)
            
            print(f"  Fold {fold_no}...", end=' ')
            
            try:
                metrics = evaluate_tflite_model(tflite_path, x_test, y_test, is_quantized, use_gpu)
                
                results.append({
                    'model': model_name,
                    'fold': fold_no,
                    'model_type': model_type,
                    'quantization': quantization,
                    'device': metrics['delegate'],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'duration_s': metrics['duration_s'],
                    'gpu_power_mean_w': metrics['gpu_power_mean_w'],
                    'gpu_power_max_w': metrics['gpu_power_max_w'],
                    'gpu_usage_mean': metrics['gpu_usage_mean'],
                    'gpu_energy_wh': metrics['gpu_energy_wh'],
                    'cpu_power_mean_w': metrics['cpu_power_mean_w'],
                    'cpu_power_max_w': metrics['cpu_power_max_w'],
                    'cpu_energy_wh': metrics['cpu_energy_wh'],
                    'energy_wh': metrics['energy_wh'],
                    'model_size_kb': metrics['model_size_kb']
                })
                
                print(f"[{metrics['delegate']}] Acc: {metrics['accuracy']:.2f}%, Energy: {metrics['energy_wh']:.6f}Wh, Size: {metrics['model_size_kb']:.1f}KB")
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*80}\n")
    
    if len(results) > 0:
        print("SUMMARY:")
        print(f"  Average Accuracy:  {df['accuracy'].mean():.2f}%")
        print(f"  Average F1 Score:  {df['f1_score'].mean():.4f}")
        print(f"  Average Energy:    {df['energy_wh'].mean():.6f}Wh")
        print(f"  Average Duration:  {df['duration_s'].mean():.2f}s")
        print(f"  Average Size:      {df['model_size_kb'].mean():.1f}KB")
        print(f"  Device Distribution:")
        print(df['device'].value_counts().to_string())
    
    return df