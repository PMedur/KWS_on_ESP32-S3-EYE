#include "inference.h"
#include "timing.h"

bool setup_tflite() {
    tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
    if (!tensor_arena) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena in PSRAM");
        return false;
    }
    
    const tflite::Model *model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model version mismatch: %" PRIu32 " vs %d", model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }
    
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddMaxPool2D();
    resolver.AddQuantize();
    resolver.AddShape();
    resolver.AddStridedSlice();
    resolver.AddPack();
    
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "Failed to allocate tensors");
        return false;
    }
    
    ESP_LOGI(TAG, "TensorFlow Lite model initialized");
    return true;
}

// Inference task
void inference_task(void *arg) {
    ESP_LOGI(TAG, "Inference task started");
    
    int8_t received_mfcc[NUM_MFCC_FEATURES * MFCC_WIDTH];
    
    while (1) {
        if (xQueueReceive(g_inference_result_queue, received_mfcc, portMAX_DELAY) != pdPASS) {
            continue;
        }

        timing_start_inference();
        
        TfLiteTensor *input_tensor = interpreter->input(0);
        memcpy(input_tensor->data.int8, received_mfcc, sizeof(received_mfcc));
        
        // Run inference
        if (interpreter->Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Failed to invoke TFLite interpreter");
            continue;
        }

        timing_end_inference();

        TfLiteTensor *output_tensor = interpreter->output(0);
        
        float dequantized_output[num_labels];
        for (int i = 0; i < num_labels; i++) {
            dequantized_output[i] = 0.00390625f * (output_tensor->data.int8[i] + 128);
        }
        
        int indices[num_labels];
        for (int i = 0; i < num_labels; i++) {
            indices[i] = i;
        }
        
        for (int i = 0; i < num_labels - 1; i++) {
            for (int j = 0; j < num_labels - i - 1; j++) {
                if (dequantized_output[indices[j]] < dequantized_output[indices[j + 1]]) {
                    int temp = indices[j];
                    indices[j] = indices[j + 1];
                    indices[j + 1] = temp;
                }
            }
        }
        
        int top_6_indices[6];
        float top_6_values[6];
        for (int i = 0; i < 6; i++) {
            top_6_indices[i] = indices[i];
            top_6_values[i] = dequantized_output[indices[i]];
        }

        ESP_LOGI(TAG, "Top prediction: %s (%.5f)", 
                 labels[top_6_indices[0]], top_6_values[0]);
        
        ESP_LOGI(TAG, "Top 6: 1.%s(%.5f) 2.%s(%.5f) 3.%s(%.5f) 4.%s(%.5f) 5.%s(%.5f) 6.%s(%.5f)",
                 labels[top_6_indices[0]], top_6_values[0],
                 labels[top_6_indices[1]], top_6_values[1],
                 labels[top_6_indices[2]], top_6_values[2],
                 labels[top_6_indices[3]], top_6_values[3],
                 labels[top_6_indices[4]], top_6_values[4],
                 labels[top_6_indices[5]], top_6_values[5]);
        
        timing_calculate_and_log();
    }
}