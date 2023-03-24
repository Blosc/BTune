#ifndef BLOSC2_EP_H
#define BLOSC2_EP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <blosc2.h>

#define ENTROPY_PROBE_ID 244
void b2ep_register_codec(blosc2_codec *codec);
#define FILTER_STOP 3
#ifdef __cplusplus
}
#endif

#endif
