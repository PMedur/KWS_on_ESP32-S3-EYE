#ifndef KEYWORD_SPOTTING_H
#define KEYWORD_SPOTTING_H

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_system.h"
#include "driver/i2s.h"
#include "driver/gpio.h"
#include <inttypes.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
#include "kiss_fft/kiss_fft.h"
#include <cmath>
#include <math.h>
#include <float.h>
#include <cstring>
#include <algorithm>
#include "esp_heap_caps.h"
#include "esp_attr.h"

// Audio configurations
#define SAMPLE_RATE_HZ          16000
#define RESAMPLED_RATE_HZ       8040
#define AUDIO_FRAME_LENGTH_MS   1000
#define SAMPLES_PER_FRAME       (SAMPLE_RATE_HZ * AUDIO_FRAME_LENGTH_MS / 1000)
#define RESAMPLED_SAMPLES       (RESAMPLED_RATE_HZ * AUDIO_FRAME_LENGTH_MS / 1000)

// MFCC parameters
#define FRAME_SIZE              200    // 25ms at 8kHz (0.025 * 8000)
#define STEP_SIZE               160    // 20ms at 8kHz (0.020 * 8000) - NO OVERLAP
#define FFT_SIZE                256    // FFT size
#define NUM_MFCC_FEATURES       13     // Number of MFCC coefficients
#define NUM_MEL_FILTERS         20     // Number of Mel filterbanks
#define MFCC_WIDTH              50     // Expected width for final 13x50 output
#define MEL_LOW_FREQ            300    // Lower cutoff frequency
#define MEL_HIGH_FREQ           4000   // Upper cutoff frequency (Nyquist at 8kHz)
#define PRE_EMPHASIS_FACTOR     0.0f   // No pre-emphasis
#define CEPLIFTER               0      // No liftering

// VAD parameters
#define VAD_FRAME_SIZE          160      
#define TARGET_CAPTURE_SAMPLES  16000     // Total samples including preroll
#define SPEECH_START_MULTIPLIER 2.0f   // Speech is 2x louder than silence
#define SPEECH_END_MULTIPLIER 1.5f     // End when drops to 1.5x silence

// ESP32-S3-EYE microphone pins
#define I2S_MIC_BCK              GPIO_NUM_41
#define I2S_MIC_WS               GPIO_NUM_42
#define I2S_MIC_DATA             GPIO_NUM_2
#define I2S_PORT_NUM             I2S_NUM_0

// TensorFlow Lite model arena size
constexpr int kTensorArenaSize = 1000 * 1024;

// Voice Activity Detector
typedef struct {
    int16_t* capture_buffer;
    size_t capture_size;
    size_t capture_capacity;
    bool is_capturing;
    bool capture_complete;
    
    float noise_floor;
    float speech_threshold;
    int silence_frames;
    int speech_frames;
    
    int16_t* preroll_buffer;
    size_t preroll_size;
    size_t preroll_write_pos;
    bool preroll_filled;
} VoiceDetector;

// Audio buffer structure
typedef struct {
    int16_t *samples;
    size_t sample_count;
    uint32_t capture_time;
    uint32_t utterance_id;
} AudioBuffer;

#define NUM_AUDIO_BUFFERS 4

// Global variables
extern const char *TAG;
extern AudioBuffer g_audio_buffers[NUM_AUDIO_BUFFERS];
extern QueueHandle_t g_audio_buffer_queue;
extern QueueHandle_t g_inference_result_queue;
extern const char *labels[];
extern const int num_labels;

// Global buffers
extern int16_t *g_audio_input_buffer;
extern int16_t *g_resampled_buffer;
extern float *g_normalized_buffer;
extern float *g_mfcc_features;
extern float *g_transposed_mfcc;
extern int8_t *g_quantized_mfcc;
extern uint8_t *tensor_arena;
extern tflite::MicroInterpreter *interpreter;
extern kiss_fft_cfg fft_cfg;
extern kiss_fft_cpx *fft_in;
extern kiss_fft_cpx *fft_out;
extern float *g_mel_filterbank;
extern float *g_window_func;
extern float *temp_frame;
extern float *fft_magnitude;
extern float *mel_output;
extern float *mfcc_output;

// Function declarations
uint32_t get_current_time_ms();
AudioBuffer* get_next_audio_buffer();
bool allocate_buffers();
bool setup_microphone();
bool setup_tflite();
void init_feature_extraction();

#endif // KEYWORD_SPOTTING_H