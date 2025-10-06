import os
import gc
import numpy as np
import ctypes
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras import backend as K

try:
    from resource_monitoring import ResourceMonitor
    RESOURCE_MONITORING_AVAILABLE = True
except ImportError:
    RESOURCE_MONITORING_AVAILABLE = False
    ResourceMonitor = None

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU available: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class MetricsHistory(Callback):
    """Track and display best validation metrics during training."""
    
    def on_train_begin(self, logs=None):
        self.best_metrics = {
            'acc': 0,
            'precision_m': 0,
            'recall_m': 0,
            'f1_m': 0,
            'epoch': 0
        }
    
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy', logs.get('val_acc', 0))
        
        if val_acc > self.best_metrics['acc']:
            self.best_metrics['acc'] = val_acc
            self.best_metrics['precision_m'] = logs.get('val_precision_m', 0)
            self.best_metrics['recall_m'] = logs.get('val_recall_m', 0)
            self.best_metrics['f1_m'] = logs.get('val_f1_m', 0)
            self.best_metrics['epoch'] = epoch + 1
            
            print("\n" + "="*80)
            print(f"NEW BEST METRICS at Epoch {epoch + 1}")
            print(f"Accuracy: {val_acc:.4f} | Precision: {self.best_metrics['precision_m']:.4f} | "
                  f"Recall: {self.best_metrics['recall_m']:.4f} | F1: {self.best_metrics['f1_m']:.4f}")
            print("="*80 + "\n")


def load_fold_data(fold_path, num_classes=36):
    """Load train, validation, and test data from fold file."""
    data = np.load(fold_path, allow_pickle=True)
    
    x_train = np.array(data['x_train_fold'])
    y_train = to_categorical(data['y_train_fold'], num_classes)
    x_val = np.array(data['x_val_fold'])
    y_val = to_categorical(data['y_val_fold'], num_classes)
    x_test = np.array(data['x_final_test'])
    y_test = to_categorical(data['y_final_test'], num_classes)
    
    print(f"Loaded fold: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)}")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def train_fold(fold_no, fold_path, model_builder, model_name, 
               epochs=30, batch_size=128, save_dir='trained_models',
               monitor_resources=False, early_stopping_patience=5):
    """Train model on a single fold with optional resource monitoring."""
    print(f"\n{'='*80}")
    print(f"TRAINING FOLD {fold_no} - {model_name}")
    print(f"{'='*80}\n")
    
    # Initialize resource monitoring
    resource_monitor = None
    if monitor_resources:
        if not RESOURCE_MONITORING_AVAILABLE:
            print("Warning: resource_monitoring.py not found, monitoring disabled")
            monitor_resources = False
        else:
            resource_monitor = ResourceMonitor(log_interval=1.0)
            resource_monitor.start()
    
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_fold_data(fold_path)
    
    model = model_builder(x_train.shape[1:])
    
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{model_name}_fold_{fold_no}_best.h5")
    
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )
    
    metrics_history = MetricsHistory()
    
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[checkpoint, early_stopping, metrics_history],
        verbose=1
    )
    
    model.load_weights(checkpoint_path)
    
    val_scores = model.evaluate(x_val, y_val, verbose=0)
    test_scores = model.evaluate(x_test, y_test, verbose=0)
    
    # Stop resource monitoring
    resource_stats = None
    if monitor_resources and resource_monitor:
        resource_stats = resource_monitor.stop()
        resource_monitor.print_stats(resource_stats)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"FOLD {fold_no} RESULTS")
    print(f"{'='*80}")
    print(f"Validation:")
    print(f"  Loss:      {val_scores[0]:.4f}")
    print(f"  Accuracy:  {val_scores[1]*100:.2f}%")
    print(f"  Precision: {metrics_history.best_metrics['precision_m']:.4f}")
    print(f"  Recall:    {metrics_history.best_metrics['recall_m']:.4f}")
    print(f"  F1 Score:  {metrics_history.best_metrics['f1_m']:.4f}")
    print(f"\nTest:")
    print(f"  Loss:      {test_scores[0]:.4f}")
    print(f"  Accuracy:  {test_scores[1]*100:.2f}%")
    print(f"  Precision: {test_scores[2]:.4f}")
    print(f"  Recall:    {test_scores[3]:.4f}")
    print(f"  F1 Score:  {test_scores[4]:.4f}")
    
    if resource_stats:
        print(f"\nResources:")
        print(f"  Duration:     {resource_stats['duration_seconds']:.1f}s ({resource_stats['duration_seconds']/60:.1f}m)")
        
        if 'cpu_energy_wh' in resource_stats:
            print(f"  CPU Power:    {resource_stats['cpu_power_mean_w']:.1f}W (max: {resource_stats['cpu_power_max_w']:.1f}W)")
            print(f"  CPU Energy:   {resource_stats['cpu_energy_wh']:.6f}Wh")
        
        if 'gpu_energy_wh' in resource_stats:
            print(f"  GPU Power:    {resource_stats['gpu_power_mean_w']:.1f}W (max: {resource_stats['gpu_power_max_w']:.1f}W)")
            print(f"  GPU Energy:   {resource_stats['gpu_energy_wh']:.3f}Wh")
        
        total_energy = resource_stats.get('cpu_energy_wh', 0) + resource_stats.get('gpu_energy_wh', 0)
        if total_energy > 0:
            print(f"  Total Energy: {total_energy:.6f}Wh")
    
    stopped_epoch = len(history.history['loss'])
    if stopped_epoch < epochs:
        print(f"\n  Early stopping triggered at epoch {stopped_epoch}/{epochs}")
    
    print(f"{'='*80}\n")
    
    # Prepare results
    results = {
        'val_loss': val_scores[0],
        'val_acc': val_scores[1] * 100,
        'val_precision': metrics_history.best_metrics['precision_m'],
        'val_recall': metrics_history.best_metrics['recall_m'],
        'val_f1': metrics_history.best_metrics['f1_m'],
        'test_loss': test_scores[0],
        'test_acc': test_scores[1] * 100,
        'test_precision': test_scores[2],
        'test_recall': test_scores[3],
        'test_f1': test_scores[4],
        'stopped_epoch': stopped_epoch
    }
    
    if resource_stats:
        results['resource_stats'] = resource_stats
    
    # Cleanup
    del model, x_train, y_train, x_val, y_val, x_test, y_test, val_scores, test_scores
    K.clear_session()
    gc.collect()
    
    return results


def cross_validate(model_builder, model_name, folds_dir='new_Data_particions',
                   num_folds=10, epochs=30, batch_size=128, save_dir='trained_models',
                   monitor_resources=False, early_stopping_patience=5):
    """Perform k-fold cross-validation training."""
    print(f"\n{'#'*80}")
    print(f"STARTING {num_folds}-FOLD CROSS-VALIDATION FOR {model_name}")
    print(f"{'#'*80}\n")
    
    all_metrics = {
        'model_name': model_name,
        'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [],
        'test_loss': [], 'test_acc': [], 'test_precision': [], 'test_recall': [], 'test_f1': [],
        'stopped_epochs': []
    }
    
    if monitor_resources:
        all_metrics['resource_stats'] = []
    
    for fold_no in range(1, num_folds + 1):
        fold_path = os.path.join(folds_dir, f"fold_{fold_no}.npz")
        
        if not os.path.exists(fold_path):
            print(f"Warning: {fold_path} not found, skipping...")
            continue
        
        metrics = train_fold(
            fold_no, fold_path, model_builder, model_name,
            epochs, batch_size, save_dir, monitor_resources, early_stopping_patience
        )
        
        for key in ['val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1',
                    'test_loss', 'test_acc', 'test_precision', 'test_recall', 'test_f1']:
            all_metrics[key].append(metrics[key])
        
        all_metrics['stopped_epochs'].append(metrics['stopped_epoch'])
        
        if monitor_resources and 'resource_stats' in metrics:
            all_metrics['resource_stats'].append(metrics['resource_stats'])
        
        # Aggressive memory cleanup
        K.clear_session()
        gc.collect()
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass
    
    save_results(model_name, all_metrics)
    print_summary(model_name, all_metrics)
    
    if monitor_resources and all_metrics.get('resource_stats'):
        print_resource_summary(model_name, all_metrics['resource_stats'])
    
    return all_metrics


def save_results(model_name, metrics, save_dir='results'):
    """Save training metrics to text file."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{model_name}_metrics.txt")
    
    with open(filepath, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"CROSS-VALIDATION RESULTS: {model_name}\n")
        f.write(f"{'='*80}\n\n")
        
        f.write("VALIDATION RESULTS PER FOLD\n")
        f.write(f"{'-'*80}\n")
        for i in range(len(metrics['val_acc'])):
            f.write(f"Fold {i+1}:\n")
            f.write(f"  Loss: {metrics['val_loss'][i]:.4f} | Acc: {metrics['val_acc'][i]:.2f}% | ")
            f.write(f"Precision: {metrics['val_precision'][i]:.4f} | ")
            f.write(f"Recall: {metrics['val_recall'][i]:.4f} | ")
            f.write(f"F1: {metrics['val_f1'][i]:.4f}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("TEST RESULTS PER FOLD\n")
        f.write(f"{'-'*80}\n")
        for i in range(len(metrics['test_acc'])):
            f.write(f"Fold {i+1}:\n")
            f.write(f"  Loss: {metrics['test_loss'][i]:.4f} | Acc: {metrics['test_acc'][i]:.2f}% | ")
            f.write(f"Precision: {metrics['test_precision'][i]:.4f} | ")
            f.write(f"Recall: {metrics['test_recall'][i]:.4f} | ")
            f.write(f"F1: {metrics['test_f1'][i]:.4f}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("AVERAGE RESULTS ACROSS ALL FOLDS\n")
        f.write(f"{'-'*80}\n")
        
        f.write("Validation:\n")
        f.write(f"  Accuracy:  {np.mean(metrics['val_acc']):.2f}% ± {np.std(metrics['val_acc']):.2f}%\n")
        f.write(f"  Loss:      {np.mean(metrics['val_loss']):.4f} ± {np.std(metrics['val_loss']):.4f}\n")
        f.write(f"  Precision: {np.mean(metrics['val_precision']):.4f} ± {np.std(metrics['val_precision']):.4f}\n")
        f.write(f"  Recall:    {np.mean(metrics['val_recall']):.4f} ± {np.std(metrics['val_recall']):.4f}\n")
        f.write(f"  F1 Score:  {np.mean(metrics['val_f1']):.4f} ± {np.std(metrics['val_f1']):.4f}\n")
        
        f.write("\nTest:\n")
        f.write(f"  Accuracy:  {np.mean(metrics['test_acc']):.2f}% ± {np.std(metrics['test_acc']):.2f}%\n")
        f.write(f"  Loss:      {np.mean(metrics['test_loss']):.4f} ± {np.std(metrics['test_loss']):.4f}\n")
        f.write(f"  Precision: {np.mean(metrics['test_precision']):.4f} ± {np.std(metrics['test_precision']):.4f}\n")
        f.write(f"  Recall:    {np.mean(metrics['test_recall']):.4f} ± {np.std(metrics['test_recall']):.4f}\n")
        f.write(f"  F1 Score:  {np.mean(metrics['test_f1']):.4f} ± {np.std(metrics['test_f1']):.4f}\n")
        
        f.write(f"{'='*80}\n")
    
    print(f"Results saved to: {filepath}")


def print_summary(model_name, metrics):
    """Print training summary to console."""
    print(f"\n{'#'*80}")
    print(f"CROSS-VALIDATION SUMMARY: {model_name}")
    print(f"{'#'*80}\n")
    
    print("AVERAGE VALIDATION METRICS:")
    print(f"  Accuracy:  {np.mean(metrics['val_acc']):.2f}% ± {np.std(metrics['val_acc']):.2f}%")
    print(f"  Precision: {np.mean(metrics['val_precision']):.4f} ± {np.std(metrics['val_precision']):.4f}")
    print(f"  Recall:    {np.mean(metrics['val_recall']):.4f} ± {np.std(metrics['val_recall']):.4f}")
    print(f"  F1 Score:  {np.mean(metrics['val_f1']):.4f} ± {np.std(metrics['val_f1']):.4f}")
    
    print("\nAVERAGE TEST METRICS:")
    print(f"  Accuracy:  {np.mean(metrics['test_acc']):.2f}% ± {np.std(metrics['test_acc']):.2f}%")
    print(f"  Precision: {np.mean(metrics['test_precision']):.4f} ± {np.std(metrics['test_precision']):.4f}")
    print(f"  Recall:    {np.mean(metrics['test_recall']):.4f} ± {np.std(metrics['test_recall']):.4f}")
    print(f"  F1 Score:  {np.mean(metrics['test_f1']):.4f} ± {np.std(metrics['test_f1']):.4f}")
    
    print(f"\n{'#'*80}\n")


def print_resource_summary(model_name, resource_stats_list):
    """Print summary of resource usage across all folds."""
    print(f"\n{'='*80}")
    print(f"RESOURCE USAGE SUMMARY: {model_name}")
    print(f"{'='*80}\n")
    
    total_duration = sum(s['duration_seconds'] for s in resource_stats_list)
    avg_cpu = np.mean([s['cpu_usage_mean'] for s in resource_stats_list])
    avg_ram = np.mean([s['ram_usage_mean'] for s in resource_stats_list])
    
    print(f"Total training time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
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