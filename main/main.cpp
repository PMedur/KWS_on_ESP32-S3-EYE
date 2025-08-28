#include "keyword_spotting.h"
#include "audio_capture.h"
#include "mfcc_processing.h"
#include "inference.h"
#include "timing.h"

extern "C" void app_main() {
    ESP_LOGI(TAG, "Starting Keyword Spotting on ESP32-S3-EYE");

    timing_init_stats();
    
    size_t psram_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    if (psram_free == 0) {
        ESP_LOGE(TAG, "PSRAM is not available. This project requires PSRAM.");
        return;
    }
    ESP_LOGI(TAG, "PSRAM available: %u bytes", psram_free);
    
    if (!allocate_buffers()) {
        ESP_LOGE(TAG, "Failed to allocate buffers");
        return;
    }

    if (!setup_tflite()) {
        ESP_LOGE(TAG, "Failed to initialize TensorFlow Lite");
        return;
    }
    
    if (!setup_microphone()) {
        ESP_LOGE(TAG, "Failed to initialize microphone");
        return;
    }
    /*
    diagnose_raw_microphone();
    vTaskDelay(pdMS_TO_TICKS(3000));
    test_microphone_levels();
    */

    init_feature_extraction();
    
    g_audio_buffer_queue = xQueueCreate(4, sizeof(AudioBuffer*));
    g_inference_result_queue = xQueueCreate(4, sizeof(int8_t) * NUM_MFCC_FEATURES * MFCC_WIDTH);
    
    if (!g_audio_buffer_queue || !g_inference_result_queue) {
        ESP_LOGE(TAG, "Failed to create queues");
        return;
    }

    xTaskCreatePinnedToCore(audio_capture_task, "audio_capture", 6144, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(audio_processing_task, "audio_processing", 4096, NULL, 3, NULL, 1);
    xTaskCreatePinnedToCore(inference_task, "inference", 4096, NULL, 4, NULL, 1);
    
    ESP_LOGI(TAG, "System initialized successfully");
    
    /*while (1) {
        vTaskDelay(pdMS_TO_TICKS(5000));
        ESP_LOGI(TAG, "Free heap: %u bytes, Free PSRAM: %u bytes", 
                 heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                 heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    }*/
}