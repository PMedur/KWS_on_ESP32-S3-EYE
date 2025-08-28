#!/usr/bin/env python3
"""
Serial Audio Capture Script
Captures audio data sent from ESP32 over serial and saves as .raw/.wav files
"""

import serial
import struct
import wave
import numpy as np
import sys
import time
from pathlib import Path

class SerialAudioCapture:
    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        
    def connect(self):
        """Connect to serial port"""
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Connected to {self.port} at {self.baudrate} baud")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def hex_to_samples(self, hex_lines):
        """Convert hex data lines to int16 samples"""
        samples = []
        for line in hex_lines:
            if line.startswith("HEX:"):
                hex_data = line[4:].strip()
                # Convert hex string to bytes
                byte_data = bytes.fromhex(hex_data)
                # Convert bytes to int16 samples (little endian)
                for i in range(0, len(byte_data), 2):
                    if i + 1 < len(byte_data):
                        sample = struct.unpack('<h', byte_data[i:i+2])[0]
                        samples.append(sample)
        return samples
    
    def csv_to_samples(self, csv_lines):
        """Convert CSV data lines to int16 samples"""
        samples = []
        for line in csv_lines:
            if line.startswith("SAMPLES:"):
                values = line[8:].strip().split(',')
                for value in values:
                    try:
                        samples.append(int(value))
                    except ValueError:
                        pass
            elif line.startswith("WAVE:"):
                values = line[5:].strip().split(',')
                for value in values:
                    try:
                        samples.append(int(value))
                    except ValueError:
                        pass
        return samples
    
    def save_as_raw(self, samples, filename):
        """Save samples as raw 16-bit PCM file"""
        with open(filename, 'wb') as f:
            for sample in samples:
                f.write(struct.pack('<h', sample))
        print(f"Saved {len(samples)} samples to {filename}")
    
    def save_as_wav(self, samples, filename, sample_rate=16000):
        """Save samples as WAV file"""
        try:
            print(f"Attempting to save WAV file: {filename}")
            print(f"Sample count: {len(samples)}, Sample rate: {sample_rate}")
            
            # Validate samples
            if not samples:
                print("❌ ERROR: No samples to save!")
                return False
                
            # Clamp samples to int16 range
            clamped_samples = []
            for sample in samples:
                if sample > 32767:
                    sample = 32767
                elif sample < -32768:
                    sample = -32768
                clamped_samples.append(sample)
            
            with wave.open(str(filename), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                
                # Convert to bytes more efficiently
                audio_data = struct.pack('<' + 'h' * len(clamped_samples), *clamped_samples)
                wav_file.writeframes(audio_data)
            
            # Verify file was created
            if filename.exists() and filename.stat().st_size > 0:
                print(f"✅ Successfully saved {len(samples)} samples to {filename}")
                print(f"   File size: {filename.stat().st_size} bytes")
                return True
            else:
                print(f"❌ ERROR: WAV file not created or is empty: {filename}")
                return False
                
        except Exception as e:
            print(f"❌ ERROR saving WAV file {filename}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def analyze_samples(self, samples, label=""):
        """Analyze captured samples"""
        if not samples:
            print(f"No samples in {label}")
            return
            
        samples_array = np.array(samples, dtype=np.int16)
        
        print(f"\n=== Analysis for {label} ===")
        print(f"Samples: {len(samples)}")
        print(f"Duration: {len(samples)/16000:.3f} seconds")
        print(f"Min: {samples_array.min()}")
        print(f"Max: {samples_array.max()}")
        print(f"Mean: {samples_array.mean():.2f}")
        print(f"RMS: {np.sqrt(np.mean(samples_array.astype(float)**2)):.2f}")
        print(f"Peak-to-peak: {samples_array.max() - samples_array.min()}")
        
        # Check for issues
        if abs(samples_array.mean()) > 1000:
            print("⚠️  WARNING: High DC offset detected!")
        
        if samples_array.max() - samples_array.min() < 1000:
            print("⚠️  WARNING: Very low signal amplitude!")
        
        zero_count = np.sum(samples_array == 0)
        if zero_count > len(samples) * 0.9:
            print("⚠️  WARNING: Too many zero samples!")
        
        print(f"Zero samples: {zero_count} ({100*zero_count/len(samples):.1f}%)")
    
    def capture_audio_data(self, timeout_seconds=60):
        """Capture audio data from serial stream"""
        if not self.serial:
            print("Not connected to serial port")
            return
        
        print(f"Listening for audio data (timeout: {timeout_seconds}s)...")
        print("Send some audio from your ESP32!")
        
        start_time = time.time()
        current_capture = None
        hex_lines = []
        csv_lines = []
        captures = {}
        
        while (time.time() - start_time) < timeout_seconds:
            try:
                line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                
                if not line:
                    continue
                    
                print(line)  # Print all lines for debugging
                
                # Detect start of audio data
                if "AUDIO_DATA_START_" in line or "AUDIO_ANALYSIS_" in line:
                    label = line.split('_')[-1].replace('===', '').strip()
                    current_capture = label
                    hex_lines = []
                    csv_lines = []
                    print(f"\n📡 Started capturing: {label}")
                
                # Collect hex data
                elif line.startswith("HEX:"):
                    hex_lines.append(line)
                
                # Collect CSV data  
                elif line.startswith(("SAMPLES:", "WAVE:")):
                    csv_lines.append(line)
                
                # Detect end of audio data
                elif ("AUDIO_DATA_END_" in line or "AUDIO_ANALYSIS_END_" in line) and current_capture:
                    # Process hex data if available
                    if hex_lines:
                        samples = self.hex_to_samples(hex_lines)
                        if samples:
                            captures[f"{current_capture}_hex"] = samples
                            print(f"✅ Captured {len(samples)} samples from hex data")
                    
                    # Process CSV data if available
                    if csv_lines:
                        samples = self.csv_to_samples(csv_lines)
                        if samples:
                            captures[f"{current_capture}_csv"] = samples
                            print(f"✅ Captured {len(samples)} samples from CSV data")
                    
                    current_capture = None
                    
            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user")
                break
            except Exception as e:
                print(f"Error reading serial: {e}")
                continue
        
        # Save all captures
        timestamp = int(time.time())
        output_dir = Path(f"esp32_audio_{timestamp}")
        output_dir.mkdir(exist_ok=True)
        
        for label, samples in captures.items():
            if samples:
                # Analyze
                self.analyze_samples(samples, label)
                
                # Save as both raw and wav
                raw_file = output_dir / f"{label}.raw"
                wav_file = output_dir / f"{label}.wav"
                
                self.save_as_raw(samples, raw_file)
                self.save_as_wav(samples, wav_file)
        
        if captures:
            print(f"\n🎉 Saved {len(captures)} audio captures to: {output_dir}")
            print("\nTo import .raw files into Audacity:")
            print("  File -> Import -> Raw Data...")
            print("  - Encoding: Signed 16-bit PCM")
            print("  - Byte order: Little-endian")
            print("  - Channels: 1 (Mono)")
            print("  - Sample rate: 16000 Hz")
        else:
            print("❌ No audio data captured")
    
    def disconnect(self):
        """Disconnect from serial port"""
        if self.serial:
            self.serial.close()
            print("Disconnected from serial port")

def main():
    if len(sys.argv) < 2:
        print("Usage: python serial_audio_capture.py <serial_port> [baudrate]")
        print("Example: python serial_audio_capture.py COM3")
        print("Example: python serial_audio_capture.py /dev/ttyUSB0 115200")
        return
    
    port = sys.argv[1]
    baudrate = int(sys.argv[2]) if len(sys.argv) > 2 else 115200
    
    capture = SerialAudioCapture(port, baudrate)
    
    if capture.connect():
        try:
            capture.capture_audio_data(timeout_seconds=120)  # 2 minute timeout
        finally:
            capture.disconnect()

if __name__ == "__main__":
    main()