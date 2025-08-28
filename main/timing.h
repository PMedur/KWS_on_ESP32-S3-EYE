#ifndef TIMING_H
#define TIMING_H

#include "esp_log.h"
#include <stdint.h>
#include <string.h>

typedef struct {
    uint32_t utterance_id;
    uint32_t audio_capture_start;   
    uint32_t audio_capture_end;     
    uint32_t preprocessing_start;    
    uint32_t preprocessing_end;     
    uint32_t inference_start;       
    uint32_t inference_end;        
    uint32_t capture_time_ms;        
    uint32_t preprocessing_time_ms;  
    uint32_t inference_time_ms;      
    uint32_t total_pipeline_ms;     
} UtteranceTiming;

extern UtteranceTiming g_current_timing;
extern const char* TIMING_TAG;

typedef struct {
    uint32_t count;
    uint32_t total_capture_ms;
    uint32_t total_preprocessing_ms;
    uint32_t total_inference_ms;
    uint32_t total_pipeline_ms;
    uint32_t min_capture_ms;
    uint32_t max_capture_ms;
    uint32_t min_preprocessing_ms;
    uint32_t max_preprocessing_ms;
    uint32_t min_inference_ms;
    uint32_t max_inference_ms;
    uint32_t min_pipeline_ms;
    uint32_t max_pipeline_ms;
} TimingStats;

extern TimingStats g_timing_stats;

void timing_init_stats(void);
void timing_start_capture(uint32_t utterance_id);
void timing_end_capture(void);
void timing_start_preprocessing(void);
void timing_end_preprocessing(void);
void timing_start_inference(void);
void timing_end_inference(void);
void timing_calculate_and_log(void);
void timing_print_stats(void);
void timing_reset_stats(void);

static inline uint32_t timing_get_time_ms(void) {
    return esp_log_timestamp();
}

#endif // TIMING_H