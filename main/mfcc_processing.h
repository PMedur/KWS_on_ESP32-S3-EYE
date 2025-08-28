#ifndef MFCC_PROCESSING_H
#define MFCC_PROCESSING_H

#include "keyword_spotting.h"

void generate_hanning_window();
void generate_mel_filterbank();
void init_feature_extraction();
void resample_audio_decimation(const int16_t *input, int16_t *output);
void resample_audio_filtered(const int16_t *input, int16_t *output);
void normalize_audio(const int16_t *input, float *output, int sample_count);
void apply_window(float *frame);
void compute_fft(const float *frame, float *output_magnitude);
void apply_mel_filterbank(const float *fft_magnitude, float *mel_output);
void compute_dct(const float *mel_output, float *mfcc_output);
void transpose_mfcc(const float *input, float *output);
void quantize_mfcc(const float *mfcc, int8_t *quantized);
void debug_mfcc_features(const float* mfcc_features, int num_frames);
void audio_processing_task(void *arg);

#endif // MFCC_PROCESSING_H