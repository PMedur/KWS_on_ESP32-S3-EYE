#include "timing.h"

const char* TIMING_TAG = "TIMING";

UtteranceTiming g_current_timing;
TimingStats g_timing_stats;

void timing_init_stats(void) {
    memset(&g_timing_stats, 0, sizeof(TimingStats));
    memset(&g_current_timing, 0, sizeof(UtteranceTiming));
    
    g_timing_stats.min_capture_ms = UINT32_MAX;
    g_timing_stats.min_preprocessing_ms = UINT32_MAX;
    g_timing_stats.min_inference_ms = UINT32_MAX;
    g_timing_stats.min_pipeline_ms = UINT32_MAX;
    
    ESP_LOGI(TIMING_TAG, "Timing statistics initialized");
}

void timing_start_capture(uint32_t utterance_id) {
    memset(&g_current_timing, 0, sizeof(UtteranceTiming));
    g_current_timing.utterance_id = utterance_id;
    g_current_timing.audio_capture_start = timing_get_time_ms();
    //ESP_LOGD(TIMING_TAG, "Started capture timing for utterance %u", utterance_id);
}

void timing_end_capture(void) {
    g_current_timing.audio_capture_end = timing_get_time_ms();
    g_current_timing.capture_time_ms = 
        g_current_timing.audio_capture_end - g_current_timing.audio_capture_start;
    //ESP_LOGD(TIMING_TAG, "Capture completed in %u ms", g_current_timing.capture_time_ms);
}

void timing_start_preprocessing(void) {
    g_current_timing.preprocessing_start = timing_get_time_ms();
    //ESP_LOGD(TIMING_TAG, "Started preprocessing timing");
}

void timing_end_preprocessing(void) {
    g_current_timing.preprocessing_end = timing_get_time_ms();
    g_current_timing.preprocessing_time_ms = 
        g_current_timing.preprocessing_end - g_current_timing.preprocessing_start;
    //ESP_LOGD(TIMING_TAG, "Preprocessing completed in %u ms", g_current_timing.preprocessing_time_ms);
}

void timing_start_inference(void) {
    g_current_timing.inference_start = timing_get_time_ms();
    //ESP_LOGD(TIMING_TAG, "Started inference timing");
}

void timing_end_inference(void) {
    g_current_timing.inference_end = timing_get_time_ms();
    g_current_timing.inference_time_ms = 
        g_current_timing.inference_end - g_current_timing.inference_start;
    
    g_current_timing.total_pipeline_ms = 
        g_current_timing.inference_end - g_current_timing.audio_capture_end;
    
    //ESP_LOGD(TIMING_TAG, "Inference completed in %u ms", g_current_timing.inference_time_ms);
}

void timing_calculate_and_log(void) {
    g_timing_stats.count++;
    g_timing_stats.total_capture_ms += g_current_timing.capture_time_ms;
    g_timing_stats.total_preprocessing_ms += g_current_timing.preprocessing_time_ms;
    g_timing_stats.total_inference_ms += g_current_timing.inference_time_ms;
    g_timing_stats.total_pipeline_ms += g_current_timing.total_pipeline_ms;
    
    if (g_current_timing.capture_time_ms < g_timing_stats.min_capture_ms)
        g_timing_stats.min_capture_ms = g_current_timing.capture_time_ms;
    if (g_current_timing.capture_time_ms > g_timing_stats.max_capture_ms)
        g_timing_stats.max_capture_ms = g_current_timing.capture_time_ms;
        
    if (g_current_timing.preprocessing_time_ms < g_timing_stats.min_preprocessing_ms)
        g_timing_stats.min_preprocessing_ms = g_current_timing.preprocessing_time_ms;
    if (g_current_timing.preprocessing_time_ms > g_timing_stats.max_preprocessing_ms)
        g_timing_stats.max_preprocessing_ms = g_current_timing.preprocessing_time_ms;
        
    if (g_current_timing.inference_time_ms < g_timing_stats.min_inference_ms)
        g_timing_stats.min_inference_ms = g_current_timing.inference_time_ms;
    if (g_current_timing.inference_time_ms > g_timing_stats.max_inference_ms)
        g_timing_stats.max_inference_ms = g_current_timing.inference_time_ms;
        
    if (g_current_timing.total_pipeline_ms < g_timing_stats.min_pipeline_ms)
        g_timing_stats.min_pipeline_ms = g_current_timing.total_pipeline_ms;
    if (g_current_timing.total_pipeline_ms > g_timing_stats.max_pipeline_ms)
        g_timing_stats.max_pipeline_ms = g_current_timing.total_pipeline_ms;
    
    ESP_LOGI(TIMING_TAG, "=== UTTERANCE %u TIMING ===", g_current_timing.utterance_id);
    ESP_LOGI(TIMING_TAG, "Capture:       %u ms", g_current_timing.capture_time_ms);
    ESP_LOGI(TIMING_TAG, "Preprocessing: %u ms", g_current_timing.preprocessing_time_ms);
    ESP_LOGI(TIMING_TAG, "Inference:     %u ms", g_current_timing.inference_time_ms);
    ESP_LOGI(TIMING_TAG, "Total Pipeline: %u ms", g_current_timing.total_pipeline_ms);
    ESP_LOGI(TIMING_TAG, "============================");
}

void timing_print_stats(void) {
    if (g_timing_stats.count == 0) {
        ESP_LOGW(TIMING_TAG, "No timing data available");
        return;
    }
    
    ESP_LOGI(TIMING_TAG, "=== TIMING STATISTICS (n=%u) ===", g_timing_stats.count);
    
    ESP_LOGI(TIMING_TAG, "CAPTURE:");
    ESP_LOGI(TIMING_TAG, "  Avg: %u ms, Min: %u ms, Max: %u ms", 
             g_timing_stats.total_capture_ms / g_timing_stats.count,
             g_timing_stats.min_capture_ms,
             g_timing_stats.max_capture_ms);
             
    ESP_LOGI(TIMING_TAG, "PREPROCESSING:");
    ESP_LOGI(TIMING_TAG, "  Avg: %u ms, Min: %u ms, Max: %u ms", 
             g_timing_stats.total_preprocessing_ms / g_timing_stats.count,
             g_timing_stats.min_preprocessing_ms,
             g_timing_stats.max_preprocessing_ms);
             
    ESP_LOGI(TIMING_TAG, "INFERENCE:");
    ESP_LOGI(TIMING_TAG, "  Avg: %u ms, Min: %u ms, Max: %u ms", 
             g_timing_stats.total_inference_ms / g_timing_stats.count,
             g_timing_stats.min_inference_ms,
             g_timing_stats.max_inference_ms);
             
    ESP_LOGI(TIMING_TAG, "TOTAL PIPELINE:");
    ESP_LOGI(TIMING_TAG, "  Avg: %u ms, Min: %u ms, Max: %u ms", 
             g_timing_stats.total_pipeline_ms / g_timing_stats.count,
             g_timing_stats.min_pipeline_ms,
             g_timing_stats.max_pipeline_ms);
             
    ESP_LOGI(TIMING_TAG, "===================================");
}

void timing_reset_stats(void) {
    timing_init_stats();
    ESP_LOGI(TIMING_TAG, "Timing statistics reset");
}