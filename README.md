# ESP32-S3 Real-Time Keyword Spotting

A high-performance, real-time keyword spotting system for ESP32-S3 microcontrollers using TensorFlow Lite for Microcontrollers. This project demonstrates on-device speech recognition capable of detecting 35 different spoken commands with low latency and efficient resource usage.

## 🎯 Project Overview

This system captures audio from a microphone, processes it through a complete MFCC (Mel-Frequency Cepstral Coefficients) feature extraction pipeline, and performs keyword classification using a quantized TensorFlow Lite neural network model. The entire pipeline runs locally on the ESP32-S3 without requiring cloud connectivity.

### Key Features

- **Real-time Processing**: Complete audio-to-prediction pipeline in <200ms
- **Voice Activity Detection (VAD)**: Intelligent speech detection with adaptive thresholds
- **MFCC Feature Extraction**: Full implementation matching `python_speech_features` library
- **35-Class Classification**: Recognizes commands like "stop", "go", "left", "right", numbers 0-9, and more
- **Memory Optimized**: Uses PSRAM efficiently with careful buffer management
- **Low Power**: Designed for battery-powered applications
- **Comprehensive Timing**: Detailed performance monitoring and statistics

## 🔧 Hardware Requirements

- **ESP32-S3-EYE** development board (or compatible ESP32-S3 with PSRAM)
- **INMP441 I2S MEMS Microphone** (or similar I2S microphone)
- **PSRAM** for buffer allocation

### Pin Configuration (ESP32-S3-EYE)
```cpp
#define I2S_MIC_BCK              GPIO_NUM_41  // Bit Clock
#define I2S_MIC_WS               GPIO_NUM_42  // Word Select
#define I2S_MIC_DATA             GPIO_NUM_2   // Data Input
```

## 📦 Dependencies

- **ESP-IDF v5.0+**
- **TensorFlow Lite for Microcontrollers**
- **KissFFT** library for Fast Fourier Transform
- Standard ESP32 components (I2S, FreeRTOS, etc.)
  
Component Dependencies
The following component need to be available in your components/ folder in the main (/) path or installed via component manager:
- tflite-micro - TensorFlow Lite for Microcontrollers

## 🏗️ Architecture

### System Components

1. **Audio Capture** (`audio_capture.cpp`)
   - I2S microphone interface at 16kHz sampling rate
   - Voice Activity Detection with adaptive noise floor estimation
   - Circular preroll buffer for capturing speech onset
   - Configurable capture duration (1-second windows)

2. **Feature Extraction** (`mfcc_processing.cpp`)
   - Audio resampling from 16kHz to 8kHz using anti-aliasing filter
   - MFCC computation with 13 coefficients across 50 time frames
   - Mel-scale filterbank with 20 filters (300-4000Hz range)
   - Hanning window and DCT transformation
   - Model-specific quantization to int8

3. **Neural Network Inference** (`inference.cpp`)
   - TensorFlow Lite Micro interpreter
   - Quantized CNN model optimized for microcontrollers
   - 700KB tensor arena in PSRAM
   - Multi-threaded pipeline for continuous processing

4. **Performance Monitoring** (`timing.cpp`)
   - Real-time timing analysis for each pipeline stage
   - Statistical tracking of latencies and throughput
   - Memory usage monitoring

### Data Flow

```
Microphone → I2S → VAD → Audio Buffer → Resampling → 
MFCC Extraction → Quantization → TF Lite Model → Classification
```

## 🚀 Getting Started

### 1. Hardware Setup
- Ensure PSRAM is enabled in your ESP32-S3 board

### 2. Build Configuration
```bash
# Configure ESP-IDF project
idf.py menuconfig

# Enable PSRAM in Component config → ESP32-specific
# Set partition table to include sufficient app space
# Enable TensorFlow Lite component
```

### 3. Compilation
```bash
idf.py build
idf.py flash monitor
```

### 4. Usage
- Power on the device
- Wait for "System initialized successfully" message
- Speak one of the supported keywords clearly
- Monitor serial output for recognition results

## 📊 Supported Keywords

The system recognizes 35 different classes:

**Commands**: stop, go, up, down, left, right, on, off, yes, no  
**Numbers**: zero, one, two, three, four, five, six, seven, eight, nine  
**Objects**: bird, cat, dog, tree, house, bed  
**Actions**: learn, follow, happy, wow  
**Names**: marvin, sheila  
**Directions**: forward, backward  
**Special**: visual

## 🔧 Configuration Options

### Voice Activity Detection
```cpp
#define SPEECH_START_MULTIPLIER 2.0f   // Sensitivity for speech detection
#define SPEECH_END_MULTIPLIER 1.5f     // Threshold for speech ending
#define TARGET_CAPTURE_SAMPLES 16000   // 1 second at 16kHz
```

### MFCC Parameters
```cpp
#define NUM_MFCC_FEATURES 13     // MFCC coefficients
#define NUM_MEL_FILTERS 20       // Mel filterbank size
#define MFCC_WIDTH 50           // Time frames
#define FRAME_SIZE 200          // 25ms frames
#define STEP_SIZE 160           // 20ms step (no overlap)
```

## 🛠️ Customization

### Adding New Keywords
1. Retrain the TensorFlow model with your custom dataset
2. Update the `labels[]` array in `globals.cpp`
3. Replace `g_model` in `model_data.h` with your quantized model
4. Adjust `num_labels` constant accordingly

### Tuning VAD Sensitivity
Modify the multiplier constants in `audio_capture.cpp` based on your acoustic environment:
- Increase `SPEECH_START_MULTIPLIER` for noisy environments
- Decrease for quiet environments or whispered speech

### Optimizing for Different Hardware
- Adjust I2S pin definitions in `keyword_spotting.h`
- Modify buffer sizes based on available PSRAM
- Tune tensor arena size for your specific model

## 📈 Monitoring and Debugging

The system provides extensive logging for development and optimization:

- **Audio diagnostics**: Raw microphone signal analysis
- **VAD status**: Real-time voice detection parameters
- **MFCC analysis**: Feature extraction statistics
- **Timing breakdowns**: Per-utterance performance metrics
- **Memory usage**: Heap and PSRAM utilization

Enable debug logging by setting ESP log level to `ESP_LOG_DEBUG`.

## 🔍 Troubleshooting

### Common Issues

1. **No audio detected**
   - Check I2S pin connections
   - Verify microphone power supply
   - Run `diagnose_raw_microphone()` function

2. **Poor recognition accuracy**
   - Ensure clean audio input (check with `test_microphone_levels()`)
   - Verify MFCC features are varying (check debug output)
   - Confirm model quantization parameters match training

3. **Memory allocation failures**
   - Ensure PSRAM is enabled and sufficient
   - Check partition table allows enough app space
   - Monitor heap usage during runtime

## 📚 Technical References

- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [ESP32-S3 Technical Reference](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/)
- [python_speech_features Documentation](https://python-speech-features.readthedocs.io/)
- [MFCC Feature Extraction Theory](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
