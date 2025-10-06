import os
import time
import psutil
import threading
import numpy as np


class ResourceMonitor:
    """Monitor CPU, GPU, and power consumption with RAPL and NVML."""
    
    def __init__(self, log_interval=1.0):
        self.log_interval = log_interval
        self.monitoring = False
        self.monitor_thread = None
        
        # Measurement storage
        self.cpu_usage = []
        self.cpu_freq = []
        self.ram_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.gpu_power = []
        self.timestamps = []
        
        # Initialize monitoring
        self.gpu_available = False
        self.device_count = 0
        self.gpu_handles = []
        self._init_gpu()
        
        self.cpu_energy_available = False
        self.rapl_domains = []
        self._init_cpu_energy()
    
    def _init_gpu(self):
        """Initialize NVIDIA GPU monitoring via pynvml."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(self.device_count):
                self.gpu_handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
            
            self.gpu_available = True
            self.pynvml = pynvml
            print(f"GPU monitoring initialized: {self.device_count} device(s) found")
            
        except ImportError:
            print("pynvml not installed. GPU monitoring disabled")
        except Exception as e:
            print(f"GPU monitoring failed: {e}")
    
    def _init_cpu_energy(self):
        """Initialize Intel RAPL for CPU energy monitoring."""
        rapl_base = '/sys/class/powercap/intel-rapl'
        
        if not os.path.exists(rapl_base):
            print("Intel RAPL not available (CPU energy monitoring disabled)")
            return
        
        try:
            for domain in os.listdir(rapl_base):
                domain_path = os.path.join(rapl_base, domain)
                energy_file = os.path.join(domain_path, 'energy_uj')
                name_file = os.path.join(domain_path, 'name')
                
                if os.path.exists(energy_file) and os.path.exists(name_file):
                    try:
                        with open(energy_file, 'r') as f:
                            f.read()
                        with open(name_file, 'r') as f:
                            name = f.read().strip()
                        
                        self.rapl_domains.append({
                            'path': energy_file,
                            'name': name,
                            'domain': domain
                        })
                    except PermissionError:
                        print(f"Permission denied for {energy_file}")
                        print("Run: sudo chmod -R a+r /sys/class/powercap/intel-rapl/")
                        return
                    except:
                        continue
            
            if self.rapl_domains:
                self.cpu_energy_available = True
                print(f"CPU energy monitoring (RAPL) initialized: {len(self.rapl_domains)} domain(s) found")
                for domain in self.rapl_domains:
                    print(f"  - {domain['name']} ({domain['domain']})")
            else:
                print("No RAPL domains found")
                
        except Exception as e:
            print(f"RAPL initialization failed: {e}")
    
    def _read_cpu_energy(self):
        """Read total CPU energy from RAPL (microjoules)."""
        if not self.cpu_energy_available:
            return 0
        
        total_energy_uj = 0
        for domain in self.rapl_domains:
            try:
                with open(domain['path'], 'r') as f:
                    total_energy_uj += int(f.read().strip())
            except:
                pass
        
        return total_energy_uj
    
    def _get_gpu_stats(self):
        """Get GPU utilization, memory, and power."""
        if not self.gpu_available:
            return None
        
        try:
            gpu_stats = []
            for handle in self.gpu_handles:
                util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                power_mw = self.pynvml.nvmlDeviceGetPowerUsage(handle)
                
                gpu_stats.append({
                    'utilization': util.gpu,
                    'memory_mb': mem.used / (1024 * 1024),
                    'power_w': power_mw / 1000.0
                })
            
            return gpu_stats
        except:
            return None
    
    def _monitor_loop(self):
        """Background monitoring thread."""
        start_cpu_energy_uj = self._read_cpu_energy()
        last_cpu_energy_uj = start_cpu_energy_uj
        last_energy_time = time.time()
        cpu_power_readings = []
        
        while self.monitoring:
            timestamp = time.time()
            
            # CPU/RAM measurements
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_frequency = psutil.cpu_freq().current if psutil.cpu_freq() else 0
            ram_percent = psutil.virtual_memory().percent
            
            # CPU power from RAPL
            current_cpu_energy_uj = self._read_cpu_energy()
            current_time = time.time()
            time_delta = current_time - last_energy_time
            
            if self.cpu_energy_available and time_delta > 0:
                energy_delta_uj = current_cpu_energy_uj - last_cpu_energy_uj
                
                # Handle counter overflow
                if energy_delta_uj < 0:
                    energy_delta_uj += 262144 * 1_000_000
                
                instant_power_w = (energy_delta_uj / 1_000_000) / time_delta
                cpu_power_readings.append(instant_power_w)
                
                last_cpu_energy_uj = current_cpu_energy_uj
                last_energy_time = current_time
            
            # GPU measurements
            gpu_stats = self._get_gpu_stats()
            
            # Store measurements
            self.timestamps.append(timestamp)
            self.cpu_usage.append(cpu_percent)
            self.cpu_freq.append(cpu_frequency)
            self.ram_usage.append(ram_percent)
            
            if gpu_stats:
                self.gpu_usage.append(np.mean([g['utilization'] for g in gpu_stats]))
                self.gpu_memory.append(np.mean([g['memory_mb'] for g in gpu_stats]))
                self.gpu_power.append(np.mean([g['power_w'] for g in gpu_stats]))
            else:
                self.gpu_usage.append(0)
                self.gpu_memory.append(0)
                self.gpu_power.append(0)
            
            time.sleep(self.log_interval)
        
        self.cpu_power_readings = cpu_power_readings
        self.final_cpu_energy_uj = self._read_cpu_energy()
        self.start_cpu_energy_uj = start_cpu_energy_uj
    
    def start(self):
        """Start monitoring in background thread."""
        if not self.monitoring:
            self.monitoring = True
            self.cpu_power_readings = []
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print("Resource monitoring started")
    
    def stop(self):
        """Stop monitoring and return statistics."""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2)
            print("Resource monitoring stopped")
            return self.get_stats()
        return None
    
    def get_stats(self):
        """Calculate statistics from monitoring session."""
        if not self.timestamps:
            return None
        
        duration = self.timestamps[-1] - self.timestamps[0]
        
        stats = {
            'duration_seconds': duration,
            'cpu_usage_mean': np.mean(self.cpu_usage),
            'cpu_usage_max': np.max(self.cpu_usage),
            'cpu_freq_mean': np.mean(self.cpu_freq),
            'ram_usage_mean': np.mean(self.ram_usage),
            'ram_usage_max': np.max(self.ram_usage)
        }
        
        # CPU energy statistics
        if self.cpu_energy_available and hasattr(self, 'final_cpu_energy_uj'):
            energy_delta_uj = self.final_cpu_energy_uj - self.start_cpu_energy_uj
            
            if energy_delta_uj < 0:
                energy_delta_uj += 262144 * 1_000_000
            
            cpu_energy_j = energy_delta_uj / 1_000_000
            
            stats.update({
                'cpu_power_mean_w': cpu_energy_j / duration if duration > 0 else 0,
                'cpu_power_max_w': np.max(self.cpu_power_readings) if self.cpu_power_readings else 0,
                'cpu_energy_j': cpu_energy_j,
                'cpu_energy_wh': cpu_energy_j / 3600
            })
        
        # GPU statistics
        if self.gpu_available and any(self.gpu_usage):
            stats.update({
                'gpu_usage_mean': np.mean(self.gpu_usage),
                'gpu_usage_max': np.max(self.gpu_usage),
                'gpu_memory_mean_mb': np.mean(self.gpu_memory),
                'gpu_memory_max_mb': np.max(self.gpu_memory),
                'gpu_power_mean_w': np.mean(self.gpu_power),
                'gpu_power_max_w': np.max(self.gpu_power),
                'gpu_energy_wh': np.mean(self.gpu_power) * duration / 3600
            })
        
        return stats
    
    def reset(self):
        """Clear all measurements."""
        self.cpu_usage = []
        self.cpu_freq = []
        self.ram_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.gpu_power = []
        self.timestamps = []
        self.cpu_power_readings = []
    
    def print_stats(self, stats=None):
        """Print formatted statistics."""
        if stats is None:
            stats = self.get_stats()
        
        if stats is None:
            print("No statistics available")
            return
        
        print(f"\n{'='*80}")
        print("RESOURCE USAGE STATISTICS")
        print(f"{'='*80}")
        print(f"Duration: {stats['duration_seconds']:.1f}s ({stats['duration_seconds']/60:.1f}m)")
        
        print(f"\nCPU:")
        print(f"  Usage (mean): {stats['cpu_usage_mean']:.1f}%")
        print(f"  Usage (max):  {stats['cpu_usage_max']:.1f}%")
        print(f"  Frequency:    {stats['cpu_freq_mean']:.0f} MHz")
        
        if 'cpu_power_mean_w' in stats:
            print(f"  Power (mean): {stats['cpu_power_mean_w']:.2f} W")
            print(f"  Power (max):  {stats['cpu_power_max_w']:.2f} W")
            print(f"  Energy:       {stats['cpu_energy_wh']:.6f} Wh ({stats['cpu_energy_j']:.2f} J)")
        
        print(f"\nRAM:")
        print(f"  Usage (mean): {stats['ram_usage_mean']:.1f}%")
        print(f"  Usage (max):  {stats['ram_usage_max']:.1f}%")
        
        if 'gpu_usage_mean' in stats:
            print(f"\nGPU:")
            print(f"  Usage (mean):  {stats['gpu_usage_mean']:.1f}%")
            print(f"  Usage (max):   {stats['gpu_usage_max']:.1f}%")
            print(f"  Memory (mean): {stats['gpu_memory_mean_mb']:.0f} MB")
            print(f"  Memory (max):  {stats['gpu_memory_max_mb']:.0f} MB")
            print(f"  Power (mean):  {stats['gpu_power_mean_w']:.1f} W")
            print(f"  Power (max):   {stats['gpu_power_max_w']:.1f} W")
            print(f"  Energy:        {stats['gpu_energy_wh']:.6f} Wh")
        
        print(f"{'='*80}\n")
    
    def shutdown(self):
        """Cleanup NVML resources."""
        if self.gpu_available:
            try:
                self.pynvml.nvmlShutdown()
            except:
                pass


def save_resource_stats(stats, filepath):
    """Save resource statistics to text file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write("RESOURCE USAGE STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Duration: {stats['duration_seconds']:.1f}s ({stats['duration_seconds']/60:.1f}m)\n\n")
        
        f.write("CPU:\n")
        f.write(f"  Usage (mean): {stats['cpu_usage_mean']:.1f}%\n")
        f.write(f"  Usage (max):  {stats['cpu_usage_max']:.1f}%\n")
        f.write(f"  Frequency:    {stats['cpu_freq_mean']:.0f} MHz\n")
        
        if 'cpu_power_mean_w' in stats:
            f.write(f"  Power (mean): {stats['cpu_power_mean_w']:.2f} W\n")
            f.write(f"  Power (max):  {stats['cpu_power_max_w']:.2f} W\n")
            f.write(f"  Energy:       {stats['cpu_energy_wh']:.6f} Wh ({stats['cpu_energy_j']:.2f} J)\n")
        
        f.write("\nRAM:\n")
        f.write(f"  Usage (mean): {stats['ram_usage_mean']:.1f}%\n")
        f.write(f"  Usage (max):  {stats['ram_usage_max']:.1f}%\n\n")
        
        if 'gpu_usage_mean' in stats:
            f.write("GPU:\n")
            f.write(f"  Usage (mean):  {stats['gpu_usage_mean']:.1f}%\n")
            f.write(f"  Usage (max):   {stats['gpu_usage_max']:.1f}%\n")
            f.write(f"  Memory (mean): {stats['gpu_memory_mean_mb']:.0f} MB\n")
            f.write(f"  Memory (max):  {stats['gpu_memory_max_mb']:.0f} MB\n")
            f.write(f"  Power (mean):  {stats['gpu_power_mean_w']:.1f} W\n")
            f.write(f"  Power (max):   {stats['gpu_power_max_w']:.1f} W\n")
            f.write(f"  Energy:        {stats['gpu_energy_wh']:.6f} Wh\n")
        
        f.write("="*80 + "\n")
    
    print(f"Resource stats saved to: {filepath}")