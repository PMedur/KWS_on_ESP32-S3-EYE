#include "mfcc_processing.h"
#include "timing.h"

void generate_hanning_window() {
    for (int i = 0; i < FRAME_SIZE; i++) {
        g_window_func[i] = 0.5f - 0.5f * cosf(2.0f * M_PI * i / (FRAME_SIZE - 1));
    }
}

void generate_mel_filterbank() {
    float mel_low = 2595.0f * log10f(1.0f + MEL_LOW_FREQ / 700.0f);
    float mel_high = 2595.0f * log10f(1.0f + MEL_HIGH_FREQ / 700.0f);
    
    float mel_points[NUM_MEL_FILTERS + 2];
    for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
        mel_points[i] = mel_low + i * (mel_high - mel_low) / (NUM_MEL_FILTERS + 1);
    }
    
    int bin_points[NUM_MEL_FILTERS + 2];
    for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
        float hz = 700.0f * (powf(10.0f, mel_points[i] / 2595.0f) - 1.0f);
        bin_points[i] = (int)floorf((FFT_SIZE + 1) * hz / RESAMPLED_RATE_HZ);
    }
    
    memset(g_mel_filterbank, 0, NUM_MEL_FILTERS * (FFT_SIZE / 2) * sizeof(float));
    
    for (int i = 0; i < NUM_MEL_FILTERS; i++) {
        float enorm = 2.0f / (bin_points[i+2] - bin_points[i]);
        
        float* filter_row = g_mel_filterbank + i * (FFT_SIZE / 2);
    
        for (int j = bin_points[i]; j < bin_points[i+1]; j++) {
            if (j < FFT_SIZE / 2) {
                filter_row[j] = (float)(j - bin_points[i]) / (bin_points[i+1] - bin_points[i]) * enorm;
            }
        }
        
        for (int j = bin_points[i+1]; j < bin_points[i+2]; j++) {
            if (j < FFT_SIZE / 2) {
                filter_row[j] = (float)(bin_points[i+2] - j) / (bin_points[i+2] - bin_points[i+1]) * enorm;
            }
        }
    }
    
    for (int i = 0; i < NUM_MEL_FILTERS; i++) {
        float sum = 0;
        float* filter_row = g_mel_filterbank + i * (FFT_SIZE / 2);
        for (int j = 0; j < FFT_SIZE / 2; j++) {
            sum += filter_row[j];
        }
        if (i == 0 || i == NUM_MEL_FILTERS - 1) {
            ESP_LOGI(TAG, "MEL filter %d sum: %.3f", i, sum);
        }
    }
}

void init_feature_extraction() {
    fft_cfg = kiss_fft_alloc(FFT_SIZE, 0, NULL, NULL);
    if (!fft_cfg) {
        ESP_LOGE(TAG, "Failed to allocate FFT configuration");
        return;
    }
    
    generate_mel_filterbank();

    generate_hanning_window();
    
    ESP_LOGI(TAG, "Feature extraction components initialized");
}

void resample_audio_decimation(const int16_t *input, int16_t *output) {
    
    for (int i = 0; i < RESAMPLED_SAMPLES; i++) {
        int src_idx = i * 2;
        
        if (src_idx + 1 < SAMPLES_PER_FRAME) {
            int32_t avg = ((int32_t)input[src_idx] + (int32_t)input[src_idx + 1]) / 2;
            output[i] = (int16_t)avg;
        } else {
            output[i] = input[src_idx];
        }
    }
}

// Simple low-pass filter coefficients for anti-aliasing
// This is a 5-tap filter designed for 2:1 decimation
void resample_audio_filtered(const int16_t *input, int16_t *output) {
    const float filter[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
    const int filter_len = 5;
    const int filter_center = 2;
    
    for (int i = 0; i < RESAMPLED_SAMPLES; i++) {
        float sum = 0.0f;
        int src_center = i * 2;
        
        for (int j = 0; j < filter_len; j++) {
            int src_idx = src_center + j - filter_center;
            
            if (src_idx < 0) src_idx = 0;
            if (src_idx >= SAMPLES_PER_FRAME) src_idx = SAMPLES_PER_FRAME - 1;
            
            sum += input[src_idx] * filter[j];
        }
        
        output[i] = (int16_t)roundf(sum);
    }
}

void normalize_audio(const int16_t *input, float *output, int sample_count) {
    for (int i = 0; i < sample_count; i++) {
        output[i] = (float)input[i] / 32768.0f; 
    }
}

void apply_window(float *frame) {
    for (int i = 0; i < FRAME_SIZE; i++) {
        frame[i] *= g_window_func[i];
    }
}

void compute_fft(const float *frame, float *output_magnitude) {
    for (int i = 0; i < FFT_SIZE; i++) {
        if (i < FRAME_SIZE) {
            fft_in[i].r = frame[i];
        } else {
            fft_in[i].r = 0.0f;
        }
        fft_in[i].i = 0.0f;
    }

    kiss_fft(fft_cfg, fft_in, fft_out);

    for (int i = 0; i < FFT_SIZE / 2; i++) {
        output_magnitude[i] = sqrtf(fft_out[i].r * fft_out[i].r + fft_out[i].i * fft_out[i].i);
    }
}

void apply_mel_filterbank(const float *fft_magnitude, float *mel_output) {
    for (int i = 0; i < NUM_MEL_FILTERS; i++) {
        mel_output[i] = 0.0f;
        float* filter_row = g_mel_filterbank + i * (FFT_SIZE / 2);
        for (int j = 0; j < FFT_SIZE / 2; j++) {
            float power = fft_magnitude[j] * fft_magnitude[j];
            mel_output[i] += power * filter_row[j];
        }

        mel_output[i] = std::max(mel_output[i], 1e-30f);
    }
}

void compute_dct(const float *mel_output, float *mfcc_output) {
    float log_mel[NUM_MEL_FILTERS];
    for (int j = 0; j < NUM_MEL_FILTERS; j++) {
        log_mel[j] = logf(mel_output[j] + 1e-10f);
    }
    
    for (int i = 0; i < NUM_MFCC_FEATURES; i++) {
        float sum = 0.0f;
        for (int j = 0; j < NUM_MEL_FILTERS; j++) {
            sum += log_mel[j] * cosf(M_PI * i * (j + 0.5f) / NUM_MEL_FILTERS);
        }
        mfcc_output[i] = sum * sqrtf(2.0f / NUM_MEL_FILTERS);
    }
}

void transpose_mfcc(const float *input, float *output) {
    for (int i = 0; i < NUM_MFCC_FEATURES; i++) {
        for (int j = 0; j < MFCC_WIDTH; j++) {
            output[i * MFCC_WIDTH + j] = input[j * NUM_MFCC_FEATURES + i];
        }
    }
}

void quantize_mfcc(const float *mfcc, int8_t *quantized) {
    const float scale = 0.35915499925613403f;
    const int zero_point = 42;
    
    //ESP_LOGI(TAG, "Using MODEL quantization: scale=%.6f, zero_point=%d", scale, zero_point);
    
    for (int i = 0; i < NUM_MFCC_FEATURES * MFCC_WIDTH; i++) {
        float q = (mfcc[i] / scale) + zero_point;
        q = std::round(q);
        q = std::max(-128.0f, std::min(127.0f, q));
        quantized[i] = static_cast<int8_t>(q);
    }
    
    // Log quantized distribution
    /*int hist[8] = {0};
    int min_q = 127, max_q = -128;
    for (int i = 0; i < NUM_MFCC_FEATURES * MFCC_WIDTH; i++) {
        min_q = std::min(min_q, (int)quantized[i]);
        max_q = std::max(max_q, (int)quantized[i]);
        int bin = (quantized[i] + 128) / 32;
        if (bin >= 0 && bin < 8) hist[bin]++;
    }*/
    
    //ESP_LOGI(TAG, "Quantized range: [%d, %d]", min_q, max_q);
    //ESP_LOGI(TAG, "Quantized distribution:");
    /*for (int i = 0; i < 8; i++) {
        int start = i * 32 - 128;
        int end = (i + 1) * 32 - 1 - 128;
        float pct = 100.0f * hist[i] / (NUM_MFCC_FEATURES * MFCC_WIDTH);
        //ESP_LOGI(TAG, "  [%d:%d] %.1f%%", start, end, pct);
    }*/
}

void debug_mfcc_features(const float* mfcc_features, int num_frames) {
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int total_values = NUM_MFCC_FEATURES * num_frames;
    
    for (int i = 0; i < total_values; i++) {
        min_val = std::min(min_val, mfcc_features[i]);
        max_val = std::max(max_val, mfcc_features[i]);
        sum += mfcc_features[i];
        sum_sq += mfcc_features[i] * mfcc_features[i];
    }
    
    float mean = sum / total_values;
    float variance = (sum_sq / total_values) - (mean * mean);
    float std_dev = sqrtf(variance);
    
    /*ESP_LOGI(TAG, "MFCC Stats: min=%.3f, max=%.3f, mean=%.3f, std=%.3f, range=%.3f", 
             min_val, max_val, mean, std_dev, max_val - min_val);*/
    
    //ESP_LOGI(TAG, "Sample MFCC values:");
    for (int frame = 0; frame < 3 && frame < num_frames; frame++) {
        char buffer[200];
        int offset = snprintf(buffer, sizeof(buffer), "Frame %d: ", frame);
        for (int coeff = 0; coeff < 5 && coeff < NUM_MFCC_FEATURES; coeff++) {
            float value = mfcc_features[frame * NUM_MFCC_FEATURES + coeff];
            offset += snprintf(buffer + offset, sizeof(buffer) - offset, "%.2f ", value);
        }
        //ESP_LOGI(TAG, "%s", buffer);
    }
    
    if (std_dev < 1.0f) {
        ESP_LOGW(TAG, "WARNING: MFCC features have very low variation (std=%.3f)", std_dev);
        ESP_LOGW(TAG, "This suggests audio processing issues!");
    }
    
    if (max_val - min_val < 5.0f) {
        ESP_LOGW(TAG, "WARNING: MFCC range is very small (%.3f)", max_val - min_val);
        ESP_LOGW(TAG, "Features may not be distinctive enough!");
    }
}

void audio_processing_task(void *arg) {
    ESP_LOGI(TAG, "Audio processing task started");
    
    while (1) {
        AudioBuffer* audio_buf = NULL;
        if (xQueueReceive(g_audio_buffer_queue, &audio_buf, portMAX_DELAY) != pdPASS) {
            continue;
        }

        timing_start_preprocessing();
        
        /*ESP_LOGI(TAG, "Processing utterance %u (%u samples)", 
                 audio_buf->utterance_id, (unsigned)audio_buf->sample_count);*/
        
        if (audio_buf->sample_count != SAMPLES_PER_FRAME) {
            /*ESP_LOGW(TAG, "WARNING: Expected %d samples but got %u", 
                     SAMPLES_PER_FRAME, (unsigned)audio_buf->sample_count);*/
            
            if (audio_buf->sample_count < SAMPLES_PER_FRAME) {
                memcpy(g_audio_input_buffer, audio_buf->samples, 
                       audio_buf->sample_count * sizeof(int16_t));
                memset(g_audio_input_buffer + audio_buf->sample_count, 0, 
                       (SAMPLES_PER_FRAME - audio_buf->sample_count) * sizeof(int16_t));
            } else {
                memcpy(g_audio_input_buffer, audio_buf->samples, 
                       SAMPLES_PER_FRAME * sizeof(int16_t));
            }
        } else {
            memcpy(g_audio_input_buffer, audio_buf->samples, 
                   SAMPLES_PER_FRAME * sizeof(int16_t));
        }
        
        resample_audio_filtered(g_audio_input_buffer, g_resampled_buffer);
        /*ESP_LOGI(TAG, "Resampled %d → %d samples (filtered decimation)", 
                 SAMPLES_PER_FRAME, RESAMPLED_SAMPLES);*/
        
        /*ESP_LOGI(TAG, "Sample check - Input[0,1,2]: %d,%d,%d → Output[0]: %d", 
                 g_audio_input_buffer[0], g_audio_input_buffer[1], 
                 g_audio_input_buffer[2], g_resampled_buffer[0]);*/
        
        normalize_audio(g_resampled_buffer, g_normalized_buffer, RESAMPLED_SAMPLES);
        
        float total_energy = 0.0f;
        for (int i = 0; i < RESAMPLED_SAMPLES; i++) {
            total_energy += g_normalized_buffer[i] * g_normalized_buffer[i];
        }
        total_energy /= RESAMPLED_SAMPLES;
        
        /*if (total_energy < 1e-10f) {
            ESP_LOGW(TAG, "WARNING: Audio appears to be all zeros! Energy=%.2e", total_energy);
        } else {
            ESP_LOGI(TAG, "Audio energy: %.6f (good signal)", total_energy);
        }*/
        
        memset(g_mfcc_features, 0, NUM_MFCC_FEATURES * MFCC_WIDTH * sizeof(float));
        
        int frames_generated = 0;
        
        for (int frame_start = 0; 
             frame_start + FRAME_SIZE <= RESAMPLED_SAMPLES && frames_generated < MFCC_WIDTH; 
             frame_start += STEP_SIZE) {

            float frame_energy = 0.0f;
            for (int j = 0; j < FRAME_SIZE; j++) {
                float sample = g_normalized_buffer[frame_start + j];
                frame_energy += sample * sample;
            }
            float log_energy = logf(frame_energy + 1e-10f);

            for (int j = 0; j < FRAME_SIZE; j++) {
                temp_frame[j] = g_normalized_buffer[frame_start + j];
            }

            apply_window(temp_frame);

            compute_fft(temp_frame, fft_magnitude);

            apply_mel_filterbank(fft_magnitude, mel_output);

            compute_dct(mel_output, mfcc_output);

            mfcc_output[0] = log_energy;

            for (int k = 0; k < NUM_MFCC_FEATURES; k++) {
                g_mfcc_features[frames_generated * NUM_MFCC_FEATURES + k] = mfcc_output[k];
            }

            /*if (frames_generated < 3) {
                ESP_LOGI(TAG, "Frame %d: Energy=%.3f, MFCC[1-5]: %.2f %.2f %.2f %.2f %.2f",
                         frames_generated, log_energy,
                         mfcc_output[1], mfcc_output[2], mfcc_output[3], 
                         mfcc_output[4], mfcc_output[5]);
            }*/
            
            frames_generated++;
        }
        
        //ESP_LOGI(TAG, "Generated %d frames from standard stepping", frames_generated);

        if (frames_generated == 49) {
            ESP_LOGI(TAG, "Duplicating last frame to reach 50 frames");
            for (int k = 0; k < NUM_MFCC_FEATURES; k++) {
                g_mfcc_features[49 * NUM_MFCC_FEATURES + k] = 
                    g_mfcc_features[48 * NUM_MFCC_FEATURES + k];
            }
            frames_generated = 50;
        }

        //debug_mfcc_features(g_mfcc_features, MFCC_WIDTH);

        float energy_min = FLT_MAX, energy_max = -FLT_MAX, energy_sum = 0;
        float mfcc_min = FLT_MAX, mfcc_max = -FLT_MAX, mfcc_sum = 0;
        
        for (int frame = 0; frame < MFCC_WIDTH; frame++) {
            float energy_val = g_mfcc_features[frame * NUM_MFCC_FEATURES + 0];
            energy_min = std::min(energy_min, energy_val);
            energy_max = std::max(energy_max, energy_val);
            energy_sum += energy_val;

            for (int coeff = 1; coeff < NUM_MFCC_FEATURES; coeff++) {
                float val = g_mfcc_features[frame * NUM_MFCC_FEATURES + coeff];
                mfcc_min = std::min(mfcc_min, val);
                mfcc_max = std::max(mfcc_max, val);
                mfcc_sum += val;
            }
        }
        
        /*ESP_LOGI(TAG, "First coefficient (log energy): min=%.3f, max=%.3f, mean=%.3f", 
                 energy_min, energy_max, energy_sum / MFCC_WIDTH);
        ESP_LOGI(TAG, "Coefficients 1-12 (MFCC): min=%.3f, max=%.3f, mean=%.3f", 
                 mfcc_min, mfcc_max, mfcc_sum / (MFCC_WIDTH * (NUM_MFCC_FEATURES - 1)));
        
        if (energy_min > -10.0f) {
            ESP_LOGW(TAG, "WARNING: Energy coefficient seems too high (min=%.3f)", energy_min);
            ESP_LOGW(TAG, "Expected negative values for quiet frames (Python shows ~-16)");
        }*/
        
        transpose_mfcc(g_mfcc_features, g_transposed_mfcc);
        
        quantize_mfcc(g_transposed_mfcc, g_quantized_mfcc);

        timing_end_preprocessing();
        
        //ESP_LOGI(TAG, "MFCC processing complete for utterance %u", audio_buf->utterance_id);

        if (xQueueSend(g_inference_result_queue, g_quantized_mfcc, 
                      100 / portTICK_PERIOD_MS) != pdPASS) {
            ESP_LOGW(TAG, "Failed to send MFCC data to inference queue");
        }
    }
}