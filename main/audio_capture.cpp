#include "audio_capture.h"
#include "timing.h"

void diagnose_raw_microphone() {
    ESP_LOGI(TAG, "=== RAW MICROPHONE DIAGNOSTIC ===");
    
    const size_t test_samples = 1000;
    int16_t* test_buffer = (int16_t*)heap_caps_malloc(test_samples * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    size_t bytes_read = 0;
    
    esp_err_t ret = i2s_read(I2S_PORT_NUM, test_buffer, test_samples * sizeof(int16_t), &bytes_read, 1000);
    
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read I2S data: %s", esp_err_to_name(ret));
        heap_caps_free(test_buffer);
        return;
    }
    
    size_t samples_read = bytes_read / sizeof(int16_t);
    
    int16_t min_val = 32767, max_val = -32768;
    int32_t sum = 0;
    int positive_count = 0, negative_count = 0, zero_count = 0;
    
    ESP_LOGI(TAG, "First 20 raw samples:");
    for (size_t i = 0; i < samples_read; i++) {
        if (i < 20) {
            printf("%d ", test_buffer[i]);
        }
        
        if (test_buffer[i] < min_val) min_val = test_buffer[i];
        if (test_buffer[i] > max_val) max_val = test_buffer[i];
        sum += test_buffer[i];
        
        if (test_buffer[i] > 0) positive_count++;
        else if (test_buffer[i] < 0) negative_count++;
        else zero_count++;
    }
    printf("\n");
    
    float avg = (float)sum / samples_read;
    
    ESP_LOGI(TAG, "Statistics for %d samples:", samples_read);
    ESP_LOGI(TAG, "  Min: %d, Max: %d", min_val, max_val);
    ESP_LOGI(TAG, "  Average (DC offset): %.1f", avg);
    ESP_LOGI(TAG, "  Positive: %d, Negative: %d, Zero: %d", 
             positive_count, negative_count, zero_count);
    ESP_LOGI(TAG, "  Peak-to-peak: %d", max_val - min_val);
    
    if (abs(avg) > 100) {
        ESP_LOGW(TAG, "Significant DC offset detected: %.1f", avg);
    }
    
    if (max_val - min_val < 100) {
        ESP_LOGW(TAG, "Very low signal amplitude! Peak-to-peak is only %d", max_val - min_val);
    }
    
    heap_caps_free(test_buffer);
}

void test_microphone_levels() {
    ESP_LOGI(TAG, "=== MICROPHONE LEVEL TEST ===");
    
    const size_t test_duration_ms = 3000;
    const size_t buffer_size = 16000 * 3;  
    int16_t* test_buffer = (int16_t*)heap_caps_malloc(buffer_size * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    
    size_t total_read = 0;
    TickType_t start_ticks = xTaskGetTickCount();
    
    while ((xTaskGetTickCount() - start_ticks) < pdMS_TO_TICKS(test_duration_ms) && 
           total_read < buffer_size * sizeof(int16_t)) {
        size_t bytes_read = 0;
        i2s_read(I2S_PORT_NUM, test_buffer + total_read/sizeof(int16_t), 
                1024, &bytes_read, 100);
        total_read += bytes_read;
    }
    
    size_t samples_read = total_read / sizeof(int16_t);
    
    for (size_t chunk = 0; chunk < samples_read; chunk += 1600) {
        size_t chunk_size = (chunk + 1600 < samples_read) ? 1600 : (samples_read - chunk);
        int16_t chunk_max = 0;
        for (size_t i = 0; i < chunk_size; i++) {
            int16_t abs_val = abs(test_buffer[chunk + i]);
            if (abs_val > chunk_max) chunk_max = abs_val;
        }
        ESP_LOGI(TAG, "100ms chunk %d: peak amplitude = %d", chunk/1600, chunk_max);
    }
    
    heap_caps_free(test_buffer);
}

void remove_dc_offset_from_buffer(int16_t* buffer, size_t num_samples, float gain) {
    int64_t sum = 0;
    for (size_t i = 0; i < num_samples; i++) {
        sum += buffer[i];
    }
    int16_t dc_offset = sum / num_samples;
    
    for (size_t i = 0; i < num_samples; i++) {
        int32_t sample = buffer[i] - dc_offset;
        sample = (int32_t)(sample * gain);
        
        if (sample > 32767) sample = 32767;
        if (sample < -32768) sample = -32768;
        
        buffer[i] = (int16_t)sample;
    }
}

VoiceDetector* init_voice_detector(void) {
    VoiceDetector* vad = (VoiceDetector*)heap_caps_calloc(1, sizeof(VoiceDetector), MALLOC_CAP_SPIRAM);
    if (!vad) return NULL;
    
    vad->capture_capacity = TARGET_CAPTURE_SAMPLES;
    vad->capture_buffer = (int16_t*)heap_caps_malloc(vad->capture_capacity * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    
    vad->noise_floor = 0.001f; 
    vad->speech_threshold = vad->noise_floor * SPEECH_START_MULTIPLIER;
    vad->silence_frames = 0;
    vad->speech_frames = 0;
    
    // Preroll buffer for 200ms of context (1600 samples at 8kHz after resampling)
    // Capturing at 16kHz, so capture 3200 samples
    vad->preroll_size = 3200;  
    vad->preroll_buffer = (int16_t*)heap_caps_malloc(vad->preroll_size * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    vad->preroll_write_pos = 0;
    vad->preroll_filled = false;  
    
    ESP_LOGI(TAG, "VAD initialized with adaptive threshold");
    return vad;
}

void free_voice_detector(VoiceDetector* vad) {
    if (vad) {
        if (vad->capture_buffer) heap_caps_free(vad->capture_buffer);
        if (vad->preroll_buffer) heap_caps_free(vad->preroll_buffer);
        heap_caps_free(vad);
    }
}

void update_preroll_buffer(VoiceDetector* vad, const int16_t* samples, size_t num_samples) {
    if (!vad->is_capturing) {
        for (size_t i = 0; i < num_samples; i++) {
            vad->preroll_buffer[vad->preroll_write_pos] = samples[i];
            vad->preroll_write_pos = (vad->preroll_write_pos + 1) % vad->preroll_size;
        }
        
        if (!vad->preroll_filled && vad->preroll_write_pos == 0) {
            vad->preroll_filled = true;
            ESP_LOGI(TAG, "Preroll buffer filled");
        }
    }
}

void copy_preroll_to_capture(VoiceDetector* vad) {
    if (!vad->preroll_filled) {
        ESP_LOGW(TAG, "Preroll buffer not filled, copying partial data");
    }
    
    size_t preroll_samples = 3200;
    if (preroll_samples > vad->preroll_size) {
        preroll_samples = vad->preroll_size;
    }
    
    size_t read_pos = (vad->preroll_write_pos + vad->preroll_size - preroll_samples) % vad->preroll_size;
    
    vad->capture_size = 0;
    for (size_t i = 0; i < preroll_samples && vad->capture_size < vad->capture_capacity; i++) {
        vad->capture_buffer[vad->capture_size++] = vad->preroll_buffer[read_pos];
        read_pos = (read_pos + 1) % vad->preroll_size;
    }
    
    //ESP_LOGI(TAG, "Copied %u preroll samples to capture buffer", (unsigned)vad->capture_size);
}

bool process_audio_vad(VoiceDetector* vad, const int16_t* samples, size_t num_samples) {
    float energy = 0.0f;
    for (size_t i = 0; i < num_samples; i++) {
        float normalized = (float)samples[i] / 32768.0f;
        energy += normalized * normalized;
    }
    energy = energy / num_samples;
    
    if (!vad->is_capturing) {
        vad->noise_floor = 0.95f * vad->noise_floor + 0.05f * energy;
        
        vad->speech_threshold = vad->noise_floor * SPEECH_START_MULTIPLIER;
        
        if (vad->speech_threshold < 0.0001f) vad->speech_threshold = 0.0001f;
        if (vad->speech_threshold > 0.01f) vad->speech_threshold = 0.01f;
        
        update_preroll_buffer(vad, samples, num_samples);
    }
    
    if (!vad->is_capturing) {
        if (energy > vad->speech_threshold) {
            vad->speech_frames++;
            
            if (vad->speech_frames >= 3) {
                ESP_LOGI(TAG, "Speech detected! Starting capture...");
                
                vad->is_capturing = true;
                vad->silence_frames = 0;

                static uint32_t next_utterance_id = 1;
                timing_start_capture(next_utterance_id);
                next_utterance_id++;
                
                copy_preroll_to_capture(vad);
            }
        } else {
            vad->speech_frames = 0; 
        }
    } else {
        size_t samples_to_copy = num_samples;
        if (vad->capture_size + samples_to_copy > vad->capture_capacity) {
            samples_to_copy = vad->capture_capacity - vad->capture_size;
        }
        
        memcpy(&vad->capture_buffer[vad->capture_size], samples, samples_to_copy * sizeof(int16_t));
        vad->capture_size += samples_to_copy;
        
        if (vad->capture_size % 1600 == 0) {
            ESP_LOGI(TAG, "Capture progress: %u/%d samples (%.1f%%)", 
                     (unsigned)vad->capture_size, SAMPLES_PER_FRAME,
                     100.0f * vad->capture_size / SAMPLES_PER_FRAME);
        }
        
        if (vad->capture_size >= SAMPLES_PER_FRAME) {
            ESP_LOGI(TAG, "Captured EXACTLY %d samples", SAMPLES_PER_FRAME);
            vad->is_capturing = false;
            vad->capture_complete = true;
            vad->capture_size = SAMPLES_PER_FRAME; 
            vad->speech_frames = 0;
            vad->silence_frames = 0;

            timing_end_capture();
            return true;
        }
    }
    
    return false;
}


void debug_vad_status(VoiceDetector* vad, float current_energy) {
    /*static int debug_counter = 0;
    if (++debug_counter % 50 == 0) {  // Every second at 20ms frames
        ESP_LOGI(TAG, "VAD Status: noise_floor=%.6f, threshold=%.6f, current=%.6f, capturing=%d",
                 vad->noise_floor, vad->speech_threshold, current_energy, vad->is_capturing);
    }*/
}

void test_vad_thresholds(VoiceDetector* vad) {
    ESP_LOGI(TAG, "=== VAD Threshold Test ===");
    ESP_LOGI(TAG, "Current settings:");
    ESP_LOGI(TAG, "  Noise floor: %.6f", vad->noise_floor);
    ESP_LOGI(TAG, "  Speech threshold: %.6f", vad->speech_threshold);
    ESP_LOGI(TAG, "  Start multiplier: %.1fx noise", SPEECH_START_MULTIPLIER);
    ESP_LOGI(TAG, "  End multiplier: %.1fx noise", SPEECH_END_MULTIPLIER);
}

void audio_capture_task(void *arg) {
    ESP_LOGI(TAG, "Audio capture task started");
    
    size_t read_buffer_size = VAD_FRAME_SIZE * 16;
    int16_t *read_buffer = (int16_t *)heap_caps_malloc(read_buffer_size * sizeof(int16_t), MALLOC_CAP_INTERNAL);
    
    if (!read_buffer) {
        ESP_LOGE(TAG, "Failed to allocate read buffer");
        vTaskDelete(NULL);
    }
    
    VoiceDetector* vad = init_voice_detector();
    if (!vad) {
        ESP_LOGE(TAG, "Failed to initialize voice detector");
        heap_caps_free(read_buffer);
        vTaskDelete(NULL);
    }
    
    uint32_t utterance_counter = 0;
    
    ESP_LOGI(TAG, "Waiting for microphone to stabilize...");
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    while (1) {
        size_t bytes_read = 0;
        esp_err_t ret = i2s_read(I2S_PORT_NUM, read_buffer, 
                                read_buffer_size * sizeof(int16_t), 
                                &bytes_read, 100 / portTICK_PERIOD_MS);
        
        if (ret != ESP_OK || bytes_read == 0) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        for (int i = 0; i < bytes_read / 4; ++i) {
            ((int16_t *) read_buffer)[i] = ((int32_t *) read_buffer)[i] >> 14;
        }
        bytes_read = bytes_read / 2;

        size_t samples_read = bytes_read / sizeof(int16_t);
        
        bool capture_complete = process_audio_vad(vad, read_buffer, samples_read);
        
        if (capture_complete) {
            utterance_counter++;
            
            ESP_LOGI(TAG, "=== UTTERANCE #%u CAPTURED ===", utterance_counter);
            ESP_LOGI(TAG, "Samples captured: %u (target was %u)", 
                    (unsigned)vad->capture_size, TARGET_CAPTURE_SAMPLES);
            
            int16_t min_val = 32767, max_val = -32768;
            for (size_t i = 0; i < vad->capture_size && i < 1000; i++) {
                if (vad->capture_buffer[i] < min_val) min_val = vad->capture_buffer[i];
                if (vad->capture_buffer[i] > max_val) max_val = vad->capture_buffer[i];
            }
            ESP_LOGI(TAG, "Captured audio range: Min=%d, Max=%d, P2P=%d", 
                    min_val, max_val, max_val - min_val);
            
            //send_vad_capture_serial(vad, utterance_counter);
            
            vad->capture_complete = false;
            
            AudioBuffer* audio_buf = get_next_audio_buffer();
            audio_buf->sample_count = vad->capture_size;
            audio_buf->capture_time = get_current_time_ms();
            audio_buf->utterance_id = utterance_counter;
            
            size_t copy_size = std::min(vad->capture_size, (size_t)SAMPLES_PER_FRAME);
            memcpy(audio_buf->samples, vad->capture_buffer, copy_size * sizeof(int16_t));
            
            if (xQueueSend(g_audio_buffer_queue, &audio_buf, 100 / portTICK_PERIOD_MS) != pdPASS) {
                ESP_LOGW(TAG, "Failed to send audio buffer to processing queue");
            } else {
                ESP_LOGI(TAG, "Sent utterance %u to processing pipeline", utterance_counter);
            }
        }
    }
    
    free_voice_detector(vad);
    heap_caps_free(read_buffer);
    vTaskDelete(NULL);
}

void send_audio_over_serial(const int16_t* samples, size_t num_samples, const char* label) {
    ESP_LOGI(TAG, "=== AUDIO_DATA_START_%s ===", label);
    ESP_LOGI(TAG, "SAMPLES: %u", (unsigned)num_samples);
    ESP_LOGI(TAG, "SAMPLE_RATE: 16000");
    ESP_LOGI(TAG, "CHANNELS: 1");
    ESP_LOGI(TAG, "FORMAT: SIGNED_16BIT_LE");
    //ESP_LOGI(TAG, "DATA_HEX_START");
    
    const uint8_t* byte_data = (const uint8_t*)samples;
    size_t total_bytes = num_samples * sizeof(int16_t);
    
    const size_t chunk_size = 32; 
    for (size_t i = 0; i < total_bytes; i += chunk_size) {
        size_t remaining = total_bytes - i;
        size_t current_chunk = (remaining < chunk_size) ? remaining : chunk_size;
        
        printf("HEX:");
        for (size_t j = 0; j < current_chunk; j++) {
            printf("%02X", byte_data[i + j]);
        }
        printf("\n");
        
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    
    ESP_LOGI(TAG, "DATA_HEX_END");
    ESP_LOGI(TAG, "=== AUDIO_DATA_END_%s ===", label);
}

void send_audio_analysis_with_sample(const int16_t* samples, size_t num_samples, const char* label) {
    ESP_LOGI(TAG, "=== AUDIO_ANALYSIS_%s ===", label);
    
    int16_t min_val = 32767, max_val = -32768;
    int64_t sum = 0;
    int64_t sum_squares = 0;
    int zero_count = 0;
    
    for (size_t i = 0; i < num_samples; i++) {
        int16_t sample = samples[i];
        
        if (sample < min_val) min_val = sample;
        if (sample > max_val) max_val = sample;
        sum += sample;
        sum_squares += (int64_t)sample * sample;
        if (sample == 0) zero_count++;
    }
    
    float mean = (float)sum / num_samples;
    float variance = ((float)sum_squares / num_samples) - (mean * mean);
    float rms = sqrtf(variance);
    
    ESP_LOGI(TAG, "STATS: samples=%u, min=%d, max=%d, mean=%.2f, rms=%.2f, zeros=%d", 
             (unsigned)num_samples, min_val, max_val, mean, rms, zero_count);
    
    size_t sample_count = (num_samples < 1000) ? num_samples : 1000;
    ESP_LOGI(TAG, "FIRST_%u_SAMPLES_START", (unsigned)sample_count);
    
    for (size_t i = 0; i < sample_count; i += 20) {
        printf("WAVE:");
        for (size_t j = 0; j < 20 && (i + j) < sample_count; j++) {
            printf("%d", samples[i + j]);
            if (j < 19 && (i + j + 1) < sample_count) printf(",");
        }
        printf("\n");
    }
    
    ESP_LOGI(TAG, "FIRST_%u_SAMPLES_END", (unsigned)sample_count);
    ESP_LOGI(TAG, "=== AUDIO_ANALYSIS_END_%s ===", label);
}

void send_vad_capture_serial(VoiceDetector* vad, uint32_t utterance_id) {
    ESP_LOGI(TAG, "send_vad_capture_serial called for utterance %u", utterance_id);
    
    if (!vad) {
        ESP_LOGW(TAG, "VAD pointer is NULL");
        return;
    }
    
    ESP_LOGI(TAG, "VAD state: capture_complete=%d, capture_size=%u", 
             vad->capture_complete, (unsigned)vad->capture_size);
    
    if (!vad->capture_complete || vad->capture_size == 0) {
        ESP_LOGW(TAG, "No VAD capture to send");
        return;
    }
    
    char label[32];
    snprintf(label, sizeof(label), "VAD_%u", utterance_id);
    
    ESP_LOGI(TAG, "Sending VAD capture %u over serial", utterance_id);
    
    send_audio_analysis_with_sample(vad->capture_buffer, vad->capture_size, label);

    ESP_LOGI(TAG, "Sending complete capture: %u samples", (unsigned)vad->capture_size);
    send_audio_over_serial(vad->capture_buffer, vad->capture_size, label);
}