#include "keyword_spotting.h"

const char *TAG = "KEYWORD_SPOTTING";

// Global buffers - allocated in PSRAM to avoid stack issues
int16_t *g_audio_input_buffer = nullptr;
int16_t *g_resampled_buffer = nullptr;
float *g_normalized_buffer = nullptr;
float *g_mfcc_features = nullptr;
float *g_transposed_mfcc = nullptr;
int8_t *g_quantized_mfcc = nullptr;

// TensorFlow Lite variables
uint8_t *tensor_arena = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;

// Queues
QueueHandle_t g_audio_buffer_queue = NULL;
QueueHandle_t g_inference_result_queue = NULL;

// Audio buffer samples
AudioBuffer g_audio_buffers[NUM_AUDIO_BUFFERS];
static int g_current_buffer_index = 0;

// FFT and feature extraction buffers - all in PSRAM
kiss_fft_cfg fft_cfg = nullptr;
kiss_fft_cpx *fft_in = nullptr;
kiss_fft_cpx *fft_out = nullptr;
float *g_mel_filterbank = nullptr;
float *g_window_func = nullptr;

// Working buffers for MFCC computation - allocated in PSRAM
float *temp_frame = nullptr;
float *fft_magnitude = nullptr;
float *mel_output = nullptr;
float *mfcc_output = nullptr;

// Labels for model output
const char *labels[] = {
    "stop", "up", "learn", "bird", "follow", "_background_noise_", "wow", "on", "marvin", "tree",
    "no", "dog", "happy", "off", "down", "six", "sheila", "bed", "seven", "visual", "four", "right",
    "five", "cat", "house", "left", "go", "eight", "forward", "one", "yes", "two", "backward", "nine", "three", "zero"
};
const int num_labels = sizeof(labels) / sizeof(labels[0]);

uint32_t get_current_time_ms() {
    return esp_log_timestamp();
}

AudioBuffer* get_next_audio_buffer() {
    AudioBuffer* buffer = &g_audio_buffers[g_current_buffer_index];
    g_current_buffer_index = (g_current_buffer_index + 1) % NUM_AUDIO_BUFFERS;
    return buffer;
}

// Allocate all buffers in PSRAM to avoid stack overflow
bool allocate_buffers() {
    ESP_LOGI(TAG, "Allocating buffers in PSRAM...");
    
    // Audio processing buffers
    g_audio_input_buffer = (int16_t*)heap_caps_malloc(SAMPLES_PER_FRAME * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    g_resampled_buffer = (int16_t*)heap_caps_malloc(RESAMPLED_SAMPLES * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    g_normalized_buffer = (float*)heap_caps_malloc(RESAMPLED_SAMPLES * sizeof(float), MALLOC_CAP_SPIRAM);
    g_mfcc_features = (float*)heap_caps_malloc(NUM_MFCC_FEATURES * MFCC_WIDTH * sizeof(float), MALLOC_CAP_SPIRAM);
    g_transposed_mfcc = (float*)heap_caps_malloc(NUM_MFCC_FEATURES * MFCC_WIDTH * sizeof(float), MALLOC_CAP_SPIRAM);
    g_quantized_mfcc = (int8_t*)heap_caps_malloc(NUM_MFCC_FEATURES * MFCC_WIDTH * sizeof(int8_t), MALLOC_CAP_SPIRAM);
    
    // FFT buffers
    fft_in = (kiss_fft_cpx*)heap_caps_malloc(FFT_SIZE * sizeof(kiss_fft_cpx), MALLOC_CAP_SPIRAM);
    fft_out = (kiss_fft_cpx*)heap_caps_malloc(FFT_SIZE * sizeof(kiss_fft_cpx), MALLOC_CAP_SPIRAM);
    
    // Feature extraction buffers
    g_mel_filterbank = (float*)heap_caps_malloc(NUM_MEL_FILTERS * (FFT_SIZE / 2) * sizeof(float), MALLOC_CAP_SPIRAM);
    g_window_func = (float*)heap_caps_malloc(FRAME_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    
    // Working buffers for MFCC computation
    temp_frame = (float*)heap_caps_malloc(FRAME_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    fft_magnitude = (float*)heap_caps_malloc((FFT_SIZE / 2) * sizeof(float), MALLOC_CAP_SPIRAM);
    mel_output = (float*)heap_caps_malloc(NUM_MEL_FILTERS * sizeof(float), MALLOC_CAP_SPIRAM);
    mfcc_output = (float*)heap_caps_malloc(NUM_MFCC_FEATURES * sizeof(float), MALLOC_CAP_SPIRAM);
    
    // Audio buffer samples
    for (int i = 0; i < NUM_AUDIO_BUFFERS; i++) {
        g_audio_buffers[i].samples = (int16_t*)heap_caps_malloc(SAMPLES_PER_FRAME * sizeof(int16_t), MALLOC_CAP_SPIRAM);
        if (!g_audio_buffers[i].samples) {
            ESP_LOGE(TAG, "Failed to allocate audio buffer %d", i);
            return false;
        }
    }
    
    if (!g_audio_input_buffer || !g_resampled_buffer || !g_normalized_buffer || 
        !g_mfcc_features || !g_transposed_mfcc || !g_quantized_mfcc || 
        !fft_in || !fft_out || !g_mel_filterbank || !g_window_func ||
        !temp_frame || !fft_magnitude || !mel_output || !mfcc_output) {
        ESP_LOGE(TAG, "Failed to allocate one or more buffers");
        return false;
    }
    
    ESP_LOGI(TAG, "All buffers allocated successfully in PSRAM");
    return true;
}

bool setup_microphone() {
    ESP_LOGI(TAG, "Setting up microphone...");
    
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 16000,
        .bits_per_sample = (i2s_bits_per_sample_t) 16,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .intr_alloc_flags = 0,
        .dma_buf_count = 3,
        .dma_buf_len = 300,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = -1,
    };
    
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_MIC_BCK,
        .ws_io_num = I2S_MIC_WS,
        .data_out_num = -1,
        .data_in_num = I2S_MIC_DATA
    };

    i2s_config.bits_per_sample = (i2s_bits_per_sample_t) 32;
    
    esp_err_t ret = i2s_driver_install(I2S_PORT_NUM, &i2s_config, 0, NULL);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to install I2S driver: %d", ret);
        return false;
    }
    
    ret = i2s_set_pin(I2S_PORT_NUM, &pin_config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to set I2S pins: %d", ret);
        return false;
    }
    
    i2s_stop(I2S_PORT_NUM);
    vTaskDelay(pdMS_TO_TICKS(100));
    i2s_start(I2S_PORT_NUM);
    
    int16_t test_buffer[64];
    size_t bytes_read = 0;
    ret = i2s_read(I2S_PORT_NUM, test_buffer, sizeof(test_buffer), &bytes_read, 500 / portTICK_PERIOD_MS);
    
    if (ret == ESP_OK && bytes_read > 0) {
        ESP_LOGI(TAG, "Microphone test successful! Read %u bytes", bytes_read);
        return true;
    } else {
        ESP_LOGE(TAG, "Microphone test failed: %d, bytes_read: %u", ret, bytes_read);
        return false;
    }
}