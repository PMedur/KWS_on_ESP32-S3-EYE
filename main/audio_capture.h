#ifndef AUDIO_CAPTURE_H
#define AUDIO_CAPTURE_H

#include "keyword_spotting.h"

void diagnose_raw_microphone();
void test_microphone_levels();
void remove_dc_offset_from_buffer(int16_t* buffer, size_t num_samples, float gain = 20.0f);
VoiceDetector* init_voice_detector();
void free_voice_detector(VoiceDetector* vad);
bool process_audio_vad(VoiceDetector* vad, const int16_t* buffer, size_t samples);
void update_preroll_buffer(VoiceDetector* vad, const int16_t* samples, size_t count);
void copy_preroll_to_capture(VoiceDetector* vad);
void debug_vad_status(VoiceDetector* vad, float current_energy);
void test_vad_thresholds(VoiceDetector* vad);
void audio_capture_task(void *arg);
void send_audio_analysis_with_sample(const int16_t* samples, size_t num_samples, const char* label);
void send_audio_over_serial(const int16_t* samples, size_t num_samples, const char* label);
void send_vad_capture_serial(VoiceDetector* vad, uint32_t utterance_id);

#endif // AUDIO_CAPTURE_H