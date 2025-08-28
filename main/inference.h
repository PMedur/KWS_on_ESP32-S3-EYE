#ifndef INFERENCE_H
#define INFERENCE_H

#include "keyword_spotting.h"

bool setup_tflite();
void inference_task(void *arg);

#endif // INFERENCE_H