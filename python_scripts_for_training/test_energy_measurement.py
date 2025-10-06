import numpy as np
import tensorflow as tf
import pandas as pd
from model_training import precision_m, recall_m, f1_m
from resource_monitoring import ResourceMonitor


def measure_test_power(model_path, fold_path, num_classes=36):
    """Measure CPU and GPU power during model inference on test set."""
    
    data = np.load(fold_path, allow_pickle=True)
    x_test = np.array(data['x_final_test'])
    y_test = tf.keras.utils.to_categorical(data['y_final_test'], num_classes)
    
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'precision_m': precision_m, 'recall_m': recall_m, 'f1_m': f1_m}
    )
    
    # Warm up
    _ = model.evaluate(x_test[:100], y_test[:100], verbose=0)
    
    # Monitor inference
    monitor = ResourceMonitor(log_interval=0.1)
    monitor.start()
    
    test_scores = model.evaluate(x_test, y_test, verbose=0)
    stats = monitor.stop()
    
    # Cleanup
    del model, x_test, y_test
    tf.keras.backend.clear_session()
    
    return {
        'test_acc': test_scores[1] * 100,
        'duration_s': stats['duration_seconds'],
        'gpu_power_mean_w': stats.get('gpu_power_mean_w', 0),
        'gpu_power_max_w': stats.get('gpu_power_max_w', 0),
        'gpu_usage_mean': stats.get('gpu_usage_mean', 0),
        'gpu_energy_wh': stats.get('gpu_energy_wh', 0),
        'cpu_power_mean_w': stats.get('cpu_power_mean_w', 0),
        'cpu_power_max_w': stats.get('cpu_power_max_w', 0),
        'cpu_energy_wh': stats.get('cpu_energy_wh', 0),
        'cpu_usage_mean': stats.get('cpu_usage_mean', 0)
    }


def measure_all_test_power(models_dir='trained_models', 
                           folds_dir='new_Data_particions',
                           output_file='test_power.csv',
                           num_models=10, num_folds=10,
                           is_finetuned=False):
    """Measure test power for all models and save to CSV."""
    
    results = []
    model_type = "Fine-tuned" if is_finetuned else "Normal"
    
    print(f"\n{'='*80}")
    print(f"MEASURING TEST POWER FOR ALL {model_type.upper()} MODELS")
    print(f"{'='*80}\n")
    
    for model_idx in range(1, num_models + 1):
        model_name = f'model_{model_idx}'
        print(f"\n{model_name}:")
        
        for fold_no in range(1, num_folds + 1):
            model_path = f'{models_dir}/{model_name}_fold_{fold_no}_{"finetuned" if is_finetuned else "best"}.h5'
            fold_path = f'{folds_dir}/fold_{fold_no}.npz'
            
            print(f"  Fold {fold_no}...", end=' ')
            
            try:
                stats = measure_test_power(model_path, fold_path)
                
                results.append({
                    'model': model_name,
                    'fold': fold_no,
                    'model_type': 'finetuned' if is_finetuned else 'normal',
                    'test_acc': stats['test_acc'],
                    'duration_s': stats['duration_s'],
                    'gpu_power_mean_w': stats['gpu_power_mean_w'],
                    'gpu_power_max_w': stats['gpu_power_max_w'],
                    'gpu_usage_mean': stats['gpu_usage_mean'],
                    'gpu_energy_wh': stats['gpu_energy_wh'],
                    'cpu_power_mean_w': stats['cpu_power_mean_w'],
                    'cpu_power_max_w': stats['cpu_power_max_w'],
                    'cpu_usage_mean': stats['cpu_usage_mean'],
                    'cpu_energy_wh': stats['cpu_energy_wh']
                })
                
                if stats['cpu_power_mean_w'] > 0:
                    print(f"CPU: {stats['cpu_power_mean_w']:.1f}W/{stats['cpu_energy_wh']:.6f}Wh | "
                          f"GPU: {stats['gpu_power_mean_w']:.1f}W/{stats['gpu_energy_wh']:.6f}Wh")
                else:
                    print(f"GPU: {stats['gpu_power_mean_w']:.1f}W/{stats['gpu_energy_wh']:.6f}Wh")
                
            except Exception as e:
                print(f"Error: {e}")
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    # Summary
    print("SUMMARY BY MODEL:")
    print(f"{'Model':<12} {'CPU (W)':<12} {'GPU (W)':<12} {'Total Energy (Wh)':<18} {'Accuracy (%)':<12}")
    print("-" * 80)
    
    for model_idx in range(1, num_models + 1):
        model_name = f'model_{model_idx}'
        model_data = df[df['model'] == model_name]
        
        if len(model_data) > 0:
            avg_cpu = model_data['cpu_power_mean_w'].mean()
            avg_gpu = model_data['gpu_power_mean_w'].mean()
            total_energy = model_data['cpu_energy_wh'].sum() + model_data['gpu_energy_wh'].sum()
            avg_acc = model_data['test_acc'].mean()
            
            print(f"{model_name:<12} {avg_cpu:<12.1f} {avg_gpu:<12.1f} {total_energy:<18.6f} {avg_acc:<12.2f}")
    
    print("\n" + "="*80)
    print("OVERALL AVERAGE:")
    print(f"  CPU Power:     {df['cpu_power_mean_w'].mean():.1f} W")
    print(f"  GPU Power:     {df['gpu_power_mean_w'].mean():.1f} W")
    print(f"  Total Energy:  {(df['cpu_energy_wh'].sum() + df['gpu_energy_wh'].sum()):.6f} Wh")
    print(f"  Avg Accuracy:  {df['test_acc'].mean():.2f}%")
    print("="*80)
    
    return df