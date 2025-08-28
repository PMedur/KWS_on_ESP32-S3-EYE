idf_component_register(SRCS 
                        "audio_capture.cpp"
                        "globals.cpp" 
                        "inference.cpp" 
                        "mfcc_processing.cpp" 
                        "main.cpp" 
                        "model_data.c" 
                        "kiss_fft/kiss_fft.c"
                        "timing.cpp"
                       INCLUDE_DIRS "." "kiss_fft"
                       PRIV_REQUIRES tflite-micro
                       REQUIRES driver freertos esp_ringbuf
                       )

target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-format)