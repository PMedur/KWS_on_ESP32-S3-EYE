import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from model_training import precision_m, recall_m, f1_m

try:
    from resource_monitoring import ResourceMonitor
    RESOURCE_MONITORING_AVAILABLE = True
except ImportError:
    RESOURCE_MONITORING_AVAILABLE = False
    ResourceMonitor = None


def load_finetuning_data(fold_path, num_classes=36):
    """
    Load fine-tuning and final test data from fold file.
    
    Args:
        fold_path: Path to fold .npz file
        num_classes: Number of classes for one-hot encoding
    
    Returns:
        (x_finetune, y_finetune), (x_final_test, y_final_test)
    """
    data = np.load(fold_path, allow_pickle=True)

    x_finetune = data['x_finetune_train']
    y_finetune = data['y_finetune_train']
    
    x_final_test = data['x_final_test']
    y_final_test = data['y_final_test']
    
    x_finetune = np.array(x_finetune) if not isinstance(x_finetune, np.ndarray) else x_finetune
    x_final_test = np.array(x_final_test) if not isinstance(x_final_test, np.ndarray) else x_final_test
    
    y_finetune = to_categorical(y_finetune, num_classes)
    y_final_test = to_categorical(y_final_test, num_classes)
    
    print(f"Loaded: Fine-tune={len(x_finetune)}, Final Test={len(x_final_test)}")
    
    return (x_finetune, y_finetune), (x_final_test, y_final_test)


def finetune_fold(fold_no, fold_path, model_path, model_name,
                  epochs=10, batch_size=128, save_dir='finetuned_models',
                  monitor_resources=False):
    """
    Fine-tune a pre-trained model on one fold.
    
    Args:
        fold_no: Fold number (1-indexed)
        fold_path: Path to fold .npz file with fine-tuning data
        model_path: Path to pre-trained model .h5 file
        model_name: Name for saving fine-tuned model
        epochs: Fine-tuning epochs
        batch_size: Batch size
        save_dir: Directory to save fine-tuned models
        monitor_resources: Enable GPU/CPU/power monitoring
    
    Returns:
        Dictionary with final test metrics
    """
    print(f"\n{'='*80}")
    print(f"FINE-TUNING FOLD {fold_no} - {model_name}")
    print(f"{'='*80}\n")
    
    resource_monitor = None
    if monitor_resources:
        if not RESOURCE_MONITORING_AVAILABLE:
            print("Warning: resource_monitoring.py not found, monitoring disabled")
            monitor_resources = False
        else:
            resource_monitor = ResourceMonitor(log_interval=1.0)
            resource_monitor.start()
    
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'precision_m': precision_m,
            'recall_m': recall_m,
            'f1_m': f1_m
        }
    )
    
    (x_finetune, y_finetune), (x_final_test, y_final_test) = load_finetuning_data(fold_path)
    
    print(f"Fine-tuning with {len(x_finetune)} samples for {epochs} epochs...")
    history = model.fit(
        x_finetune, y_finetune,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    test_scores = model.evaluate(x_final_test, y_final_test, verbose=0)
    
    resource_stats = None
    if monitor_resources and resource_monitor:
        resource_stats = resource_monitor.stop()
        resource_monitor.print_stats(resource_stats)
    
    print(f"\n{'='*80}")
    print(f"FOLD {fold_no} FINE-TUNING RESULTS")
    print(f"{'='*80}")
    print(f"Final Test - Loss: {test_scores[0]:.4f}, Acc: {test_scores[1]*100:.2f}%")
    print(f"Precision: {test_scores[2]:.4f}, Recall: {test_scores[3]:.4f}, F1: {test_scores[4]:.4f}")
    print(f"{'='*80}\n")
    
    os.makedirs(save_dir, exist_ok=True)
    finetuned_path = os.path.join(save_dir, f"{model_name}_fold_{fold_no}_finetuned.h5")
    model.save(finetuned_path)
    print(f"Saved fine-tuned model to: {finetuned_path}")
    
    results = {
        'test_loss': test_scores[0],
        'test_acc': test_scores[1] * 100,
        'test_precision': test_scores[2],
        'test_recall': test_scores[3],
        'test_f1': test_scores[4]
    }
    
    if resource_stats:
        results['resource_stats'] = resource_stats
    
    return results


def finetune_all_folds(model_name, folds_dir='new_Data_particions',
                       models_dir='trained_models', num_folds=10,
                       epochs=10, batch_size=128, save_dir='finetuned_models',
                       monitor_resources=False):
    """
    Fine-tune a model across all folds.
    
    Args:
        model_name: Name of the model to fine-tune
        folds_dir: Directory containing fold .npz files
        models_dir: Directory containing pre-trained models
        num_folds: Number of folds
        epochs: Fine-tuning epochs
        batch_size: Batch size
        save_dir: Directory to save fine-tuned models
        monitor_resources: Enable GPU/CPU/power monitoring
    
    Returns:
        Dictionary with metrics across all folds
    """
    print(f"\n{'#'*80}")
    print(f"FINE-TUNING {model_name} ACROSS {num_folds} FOLDS")
    print(f"{'#'*80}\n")
    
    all_metrics = {
        'model_name': model_name,
        'test_loss': [],
        'test_acc': [],
        'test_precision': [],
        'test_recall': [],
        'test_f1': []
    }
    
    if monitor_resources:
        all_metrics['resource_stats'] = []
    
    for fold_no in range(1, num_folds + 1):
        fold_path = os.path.join(folds_dir, f"fold_{fold_no}.npz")
        model_path = os.path.join(models_dir, f"{model_name}_fold_{fold_no}_best.h5")
        
        if not os.path.exists(fold_path):
            print(f"Warning: {fold_path} not found, skipping...")
            continue
        
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found, skipping...")
            continue
        
        metrics = finetune_fold(
            fold_no, fold_path, model_path, model_name,
            epochs, batch_size, save_dir, monitor_resources
        )
        
        for key in ['test_loss', 'test_acc', 'test_precision', 'test_recall', 'test_f1']:
            all_metrics[key].append(metrics[key])
        
        if monitor_resources and 'resource_stats' in metrics:
            all_metrics['resource_stats'].append(metrics['resource_stats'])
        
        tf.keras.backend.clear_session()
    
    save_finetuning_results(model_name, all_metrics)
    print_finetuning_summary(model_name, all_metrics)
    
    if monitor_resources and all_metrics.get('resource_stats'):
        print_finetuning_resource_summary(model_name, all_metrics['resource_stats'])
    
    return all_metrics


def save_finetuning_results(model_name, metrics, save_dir='results'):
    """Save fine-tuning results to file."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{model_name}_finetuning_metrics.txt")
    
    with open(filepath, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"FINE-TUNING RESULTS: {model_name}\n")
        f.write(f"{'='*80}\n\n")
        
        # Per-fold results
        f.write("FINAL TEST RESULTS PER FOLD (After Fine-tuning)\n")
        f.write(f"{'-'*80}\n")
        for i in range(len(metrics['test_acc'])):
            f.write(f"Fold {i+1}:\n")
            f.write(f"  Loss: {metrics['test_loss'][i]:.4f} | Acc: {metrics['test_acc'][i]:.2f}% | ")
            f.write(f"Precision: {metrics['test_precision'][i]:.4f} | ")
            f.write(f"Recall: {metrics['test_recall'][i]:.4f} | ")
            f.write(f"F1: {metrics['test_f1'][i]:.4f}\n")
        
        # Average results
        f.write(f"\n{'='*80}\n")
        f.write("AVERAGE RESULTS ACROSS ALL FOLDS\n")
        f.write(f"{'-'*80}\n")
        f.write(f"Accuracy:  {np.mean(metrics['test_acc']):.2f}% ± {np.std(metrics['test_acc']):.2f}%\n")
        f.write(f"Loss:      {np.mean(metrics['test_loss']):.4f} ± {np.std(metrics['test_loss']):.4f}\n")
        f.write(f"Precision: {np.mean(metrics['test_precision']):.4f} ± {np.std(metrics['test_precision']):.4f}\n")
        f.write(f"Recall:    {np.mean(metrics['test_recall']):.4f} ± {np.std(metrics['test_recall']):.4f}\n")
        f.write(f"F1 Score:  {np.mean(metrics['test_f1']):.4f} ± {np.std(metrics['test_f1']):.4f}\n")
        f.write(f"{'='*80}\n")
    
    print(f"Fine-tuning results saved to: {filepath}")


def print_finetuning_summary(model_name, metrics):
    """Print fine-tuning summary to console."""
    print(f"\n{'#'*80}")
    print(f"FINE-TUNING SUMMARY: {model_name}")
    print(f"{'#'*80}\n")
    
    print("AVERAGE FINAL TEST METRICS (After Fine-tuning):")
    print(f"  Accuracy:  {np.mean(metrics['test_acc']):.2f}% ± {np.std(metrics['test_acc']):.2f}%")
    print(f"  Precision: {np.mean(metrics['test_precision']):.4f} ± {np.std(metrics['test_precision']):.4f}")
    print(f"  Recall:    {np.mean(metrics['test_recall']):.4f} ± {np.std(metrics['test_recall']):.4f}")
    print(f"  F1 Score:  {np.mean(metrics['test_f1']):.4f} ± {np.std(metrics['test_f1']):.4f}")
    
    print(f"\n{'#'*80}\n")


def print_finetuning_resource_summary(model_name, resource_stats_list):
    """Print summary of resource usage across all fine-tuning folds."""
    print(f"\n{'='*80}")
    print(f"FINE-TUNING RESOURCE USAGE SUMMARY: {model_name}")
    print(f"{'='*80}\n")
    
    total_duration = sum(s['duration_seconds'] for s in resource_stats_list)
    avg_cpu = np.mean([s['cpu_usage_mean'] for s in resource_stats_list])
    avg_ram = np.mean([s['ram_usage_mean'] for s in resource_stats_list])
    
    print(f"Total fine-tuning time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Average CPU usage: {avg_cpu:.1f}%")
    print(f"Average RAM usage: {avg_ram:.1f}%")
    
    if 'gpu_usage_mean' in resource_stats_list[0]:
        avg_gpu = np.mean([s['gpu_usage_mean'] for s in resource_stats_list])
        avg_power = np.mean([s['gpu_power_mean_w'] for s in resource_stats_list])
        total_energy = sum(s['gpu_energy_wh'] for s in resource_stats_list)
        
        print(f"Average GPU usage: {avg_gpu:.1f}%")
        print(f"Average GPU power: {avg_power:.1f} W")
        print(f"Total energy consumed: {total_energy:.3f} Wh ({total_energy/1000:.6f} kWh)")
    
    print(f"{'='*80}\n")


def compare_before_after_finetuning(original_metrics, finetuned_metrics):
    """
    Compare model performance before and after fine-tuning.
    
    Args:
        original_metrics: Metrics from initial training (from cross_validate)
        finetuned_metrics: Metrics from fine-tuning (from finetune_all_folds)
    """
    print(f"\n{'='*100}")
    print(f"COMPARISON: {original_metrics['model_name']} - BEFORE vs AFTER FINE-TUNING")
    print(f"{'='*100}\n")
    
    print(f"{'Metric':<20} {'Before (Test)':<20} {'After (Fine-tuned)':<20} {'Improvement':<20}")
    print(f"{'-'*100}")
    
    metrics_to_compare = [
        ('Accuracy', 'test_acc', '%'),
        ('Precision', 'test_precision', ''),
        ('Recall', 'test_recall', ''),
        ('F1 Score', 'test_f1', '')
    ]
    
    for metric_name, key, unit in metrics_to_compare:
        before = np.mean(original_metrics[key])
        after = np.mean(finetuned_metrics[key])
        improvement = after - before
        
        before_str = f"{before:.2f}{unit}" if unit == '%' else f"{before:.4f}"
        after_str = f"{after:.2f}{unit}" if unit == '%' else f"{after:.4f}"
        
        if unit == '%':
            improvement_str = f"{improvement:+.2f}%"
        else:
            improvement_str = f"{improvement:+.4f}"
        
        print(f"{metric_name:<20} {before_str:<20} {after_str:<20} {improvement_str:<20}")
    
    print(f"{'='*100}\n")
    
    if 'resource_stats' in original_metrics and 'resource_stats' in finetuned_metrics:
        if 'gpu_energy_wh' in original_metrics['resource_stats'][0]:
            train_energy = sum(s['gpu_energy_wh'] for s in original_metrics['resource_stats'])
            finetune_energy = sum(s['gpu_energy_wh'] for s in finetuned_metrics['resource_stats'])
            
            print("ENERGY COMPARISON:")
            print(f"  Training energy:    {train_energy:.3f} Wh")
            print(f"  Fine-tuning energy: {finetune_energy:.3f} Wh")
            print(f"  Total energy:       {train_energy + finetune_energy:.3f} Wh")
            print()
