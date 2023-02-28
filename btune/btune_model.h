#ifndef BTUNE_MODEL_H
#define BTUNE_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

int btune_model_inference(blosc2_context * ctx, int * compcode, uint8_t * filter);

#ifdef __cplusplus
}
#endif

#endif  /* BTUNE_MODEL_H */
