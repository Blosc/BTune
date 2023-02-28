/**********************************************************************
  Optimize Blosc2 parameters using deep/machine learning.

  Copyright (c) 2023 The Blosc Developers <blosc@blosc.org>
  License: GNU Affero General Public License v3.0

  See COPYING.txt for details about copyright and rights to use.
***********************************************************************/

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "btune.h"
#include "btune_model.h"


// Disable shufflesize and blocksize
#define BTUNE_DISABLE_SHUFFLESIZE  true
#define BTUNE_DISABLE_BLOCKSIZE    true
#define BTUNE_DISABLE_MEMCPY       true
#define BTUNE_DISABLE_THREADS      true


// Internal btune control behaviour constants.
enum {
  BTUNE_KB = 1024,
  MAX_CODECS = 8, // TODO remove when is included in blosc.h
  NUM_FILTERS = 3, // nofilter, shuffle and bitshuffle
  NUM_SPLITS = 2, // split and nosplit
  MAX_CLEVEL = 9,
  MIN_BLOCK = 16 * BTUNE_KB,  // TODO remove when included in blosc.h
  MAX_BLOCK = 2 * BTUNE_KB * BTUNE_KB,
  MIN_BITSHUFFLE = 1,
  MIN_SHUFFLE = 2,
  MAX_SHUFFLE = 16,
  MIN_THREADS = 1,
  SOFT_STEP_SIZE = 1,
  HARD_STEP_SIZE = 2,
  MAX_STATE_THREADS = 50,  // 50 magic number big enough to not tune threads this number of times
};

static const cparams_btune cparams_btune_default = {
  BLOSC_LZ4, BLOSC_SHUFFLE, BLOSC_ALWAYS_SPLIT, 9, 0, 0, 0, 0, false, true, true, false, 100, 1.1, 100, 100
};

// Get the codecs list for btune
static codec_list * btune_get_codecs(btune_struct * btune) {
  const char * all_codecs = blosc2_list_compressors();
  codec_list * codecs = malloc(sizeof(codec_list));
  codecs->list = malloc(MAX_CODECS * sizeof(int));
  int i = 0;
  if (btune->config.comp_mode == BTUNE_COMP_HCR) {
    // In HCR mode only try with ZSTD and ZLIB
    if (strstr(all_codecs, "zstd") != NULL) {
      codecs->list[i++] = BLOSC_ZSTD;
    }
    if (strstr(all_codecs, "zlib") != NULL) {
      codecs->list[i++] = BLOSC_ZLIB;
    }
    // And disable LZ4HC as it compress typically less
    // codecs->list[i++] = BLOSC_LZ4HC;
  } else {
    // In all other modes, LZ4 is mandatory
    codecs->list[i++] = BLOSC_LZ4;
    if (btune->config.comp_mode == BTUNE_COMP_BALANCED) {
      // In BALANCE mode give BLOSCLZ a chance
      codecs->list[i++] = BLOSC_BLOSCLZ;
    }
    if (btune->config.perf_mode == BTUNE_PERF_DECOMP) {
      codecs->list[i++] = BLOSC_LZ4HC;
    }
  }
  codecs->size = i;
  return codecs;
}

static void add_codec(codec_list * codecs, int compcode) {
  for (int i = 0; i < codecs->size; i++) {
    if (codecs->list[i] == compcode) {
      return;
    }
  }
  codecs->list[codecs->size] = compcode;
  codecs->size++;
}

// Extract the cparams_btune inside blosc2_context
static void extract_btune_cparams(blosc2_context * context, cparams_btune * cparams){
  cparams->compcode = context->compcode;
  cparams->filter = context->filters[BLOSC2_MAX_FILTERS - 1];
  cparams->clevel = context->clevel;
  cparams->splitmode = context->splitmode;
  cparams->blocksize = context->blocksize;
  cparams->shufflesize = context->typesize;
  cparams->nthreads_comp = context->nthreads;
  btune_struct * btune = context->btune;
  if (btune->dctx == NULL) {
    cparams->nthreads_decomp = btune->nthreads_decomp;
  } else {
    cparams->nthreads_decomp = btune->dctx->nthreads;
  }
}

// Check if btune can still modify the clevel or has to change the direction
static bool has_ended_clevel(btune_struct * btune) {
  return ((btune->best->increasing_clevel &&
          (btune->best->clevel >= (MAX_CLEVEL - btune->step_size))) ||
          (!btune->best->increasing_clevel &&
          (btune->best->clevel <= (1 + btune->step_size))));
}

// Check if btune can still modify the shufflesize or has to change the direction
static bool has_ended_shuffle(cparams_btune * best) {
  int min_shuffle = (best->filter == BLOSC_SHUFFLE) ? MIN_SHUFFLE: MIN_BITSHUFFLE;
  return ((best->increasing_shuffle && (best->shufflesize == MAX_SHUFFLE)) ||
          (!best->increasing_shuffle && (best->shufflesize == min_shuffle)));
}

// Check if btune can still modify the nthreads or has to change the direction
static bool has_ended_threads(btune_struct * btune) {
  cparams_btune * best = btune->best;
  int nthreads;
  if (btune->threads_for_comp) {
    nthreads = best->nthreads_comp;
  } else {
    nthreads = best->nthreads_decomp;
  }
  return ((best->increasing_nthreads && (nthreads == btune->max_threads)) ||
          (!best->increasing_nthreads && (nthreads == MIN_THREADS)));
}

// Check if btune can still modify the blocksize or has to change the direction
static bool has_ended_blocksize(blosc2_context * ctx){
  btune_struct * btune = (btune_struct*) ctx->btune;
  cparams_btune * best = btune->best;
  return ((best->increasing_block &&
          ((best->blocksize > (MAX_BLOCK >> btune->step_size)) ||
          (best->blocksize > (ctx->sourcesize >> btune->step_size)))) ||
          (!best->increasing_block &&
          (best->blocksize < (MIN_BLOCK << btune->step_size))));
}

// Init a soft readapt
static void init_soft(btune_struct * btune) {
  if (has_ended_clevel(btune)) {
    btune->best->increasing_clevel = !btune->best->increasing_clevel;
  }
  btune->state = CLEVEL;
  btune->step_size = SOFT_STEP_SIZE;
  btune->readapt_from = SOFT;
}

// Init a hard readapt
static void init_hard(btune_struct * btune) {
  btune->state = CODEC_FILTER;
  btune->step_size = HARD_STEP_SIZE;
  btune->readapt_from = HARD;
  if (btune->config.perf_mode == BTUNE_PERF_DECOMP) {
    btune->threads_for_comp = false;
  } else {
    btune->threads_for_comp = true;
  }
  if (has_ended_shuffle(btune->best)) {
    btune->best->increasing_shuffle = !btune->best->increasing_shuffle;
  }
}

// Init when the number of hard is 0
static void init_without_hards(blosc2_context * ctx) {
  btune_struct * btune = (btune_struct*) ctx->btune;
  btune_behaviour behaviour = btune->config.behaviour;
  int minimum_hards = 0;
  if (!btune->config.cparams_hint) {
    minimum_hards++;
  }
  switch (behaviour.repeat_mode) {
    case BTUNE_REPEAT_ALL:
      if (behaviour.nhards_before_stop > (uint32_t)minimum_hards) {
        init_hard(btune);
        break;
      }
    case BTUNE_REPEAT_SOFT:
      if (behaviour.nsofts_before_hard > 0) {
        init_soft(btune);
        break;
      }
    case BTUNE_STOP:
      if ((minimum_hards == 0) && (behaviour.nsofts_before_hard > 0)) {
        init_soft(btune);
      } else {
        btune->state = STOP;
        btune->readapt_from = WAIT;
      }
      break;
    default:
      fprintf(stderr, "WARNING: stop mode unknown\n");
  }
  btune->is_repeating = true;
}

static const char* stcode_to_stname(btune_struct * btune) {
  switch (btune->state) {
    case CODEC_FILTER:
      return "CODEC_FILTER";
    case THREADS:
      if (btune->threads_for_comp) {
        return "THREADS_COMP";
      } else {
        return "THREADS_DECOMP";
      }
    case SHUFFLE_SIZE:
      return "SHUFFLE_SIZE";
    case CLEVEL:
      return "CLEVEL";
    case BLOCKSIZE:
      return "BLOCKSIZE";
    case MEMCPY:
      return "MEMCPY";
    case WAITING:
      return "WAITING";
    case STOP:
      return "STOP";
    default:
      return "UNKNOWN";
  }
}

static const char* readapt_to_str(readapt_type readapt) {
  switch (readapt) {
    case HARD:
      return "HARD";
    case SOFT:
      return "SOFT";
    case WAIT:
      return "WAIT";
    default:
      return "UNKNOWN";
  }
}

static const char* perf_mode_to_str(btune_performance_mode perf_mode) {
  switch (perf_mode) {
    case BTUNE_PERF_DECOMP:
      return "DECOMP";
    case BTUNE_PERF_BALANCED:
      return "BALANCED";
    case BTUNE_PERF_COMP:
      return "COMP";
    default:
      return "UNKNOWN";
  }
}

static const char* comp_mode_to_str(btune_comp_mode comp_mode) {
  switch (comp_mode) {
    case BTUNE_COMP_HSP:
      return "HSP";
    case BTUNE_COMP_BALANCED:
      return "BALANCED";
    case BTUNE_COMP_HCR:
      return "HCR";
    default:
      return "UNKNOWN";
  }
}

static void bandwidth_to_str(char * str, uint32_t bandwidth) {
  if (bandwidth < BTUNE_MBPS) {
    sprintf(str, "%d KB/s", bandwidth);
  } else if (bandwidth < BTUNE_GBPS) {
    sprintf(str, "%d MB/s", bandwidth / BTUNE_KB);
  } else if (bandwidth < BTUNE_TBPS) {
    sprintf(str, "%d GB/s", bandwidth / BTUNE_KB / BTUNE_KB);
  } else {
    sprintf(str, "%d TB/s", bandwidth / BTUNE_KB / BTUNE_KB / BTUNE_KB);
  }
}

static const char* repeat_mode_to_str(btune_repeat_mode repeat_mode) {
  switch (repeat_mode) {
    case BTUNE_REPEAT_ALL:
      return "REPEAT_ALL";
    case BTUNE_REPEAT_SOFT:
      return "REPEAT_SOFT";
    case BTUNE_STOP:
      return "STOP";
    default:
      return "UNKNOWN";
  }
}

// Init btune_struct inside blosc2_context
void btune_init(void * cfg, blosc2_context * cctx, blosc2_context * dctx) {
  btune_config * config = (btune_config *)cfg;

  // TODO CHECK CONFIG ENUMS (bandwidth range...)
  btune_struct * btune = calloc(sizeof(btune_struct), 1);
  if (config == NULL) {
    memcpy(&btune->config, &BTUNE_CONFIG_DEFAULTS, sizeof(btune_config));
  } else {
    memcpy(&btune->config, config, sizeof(btune_config));
  }

  char* envvar = getenv("BTUNE_LOG");
  if (envvar != NULL) {
    printf("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n");
    char bandwidth_str[12];
    bandwidth_to_str(bandwidth_str, btune->config.bandwidth);
    printf("BTune version: %s.\n"
                   "Perfomance Mode: %s, Compression Mode: %s, Bandwidth: %s.\n"
                   "Behaviour: Waits - %d, Softs - %d, Hards - %d, Repeat Mode - %s.\n",
           BTUNE_VERSION_STRING, perf_mode_to_str(btune->config.perf_mode),
           comp_mode_to_str(btune->config.comp_mode),
           bandwidth_str,
           btune->config.behaviour.nwaits_before_readapt,
           btune->config.behaviour.nsofts_before_hard,
           btune->config.behaviour.nhards_before_stop,
           repeat_mode_to_str(btune->config.behaviour.repeat_mode));
    printf("|    Codec   | Filter | Split | C.Level | Blocksize | Shufflesize | C.Threads | D.Threads |   Score   |  C.Ratio   |"
           "   BTune State   | Readapt | Winner\n");
  }

  btune->dctx = dctx;
  btune->codecs = btune_get_codecs(btune);

  // State attributes
  btune->rep_index = 0;
  // We want to iterate 3x per filter (NOSHUFFLE/SHUFFLE/BITSHUFFLE) and 2x per split/nonsplit
  btune->filter_split_limit = NUM_FILTERS * NUM_SPLITS;
  btune->aux_index = 0;
  btune->steps_count = 0;
  btune->nsofts = 0;
  btune->nhards = 0;
  btune->nwaitings = 0;
  btune->is_repeating = false;
  cctx->btune = btune;

  // Initial compression parameters
  cparams_btune * best = malloc(sizeof(cparams_btune));
  *best = cparams_btune_default;
  btune->best = best;
  cparams_btune * aux = malloc(sizeof(cparams_btune));
  *aux = cparams_btune_default;
  btune->aux_cparams = aux;
  best->compcode = btune->codecs->list[0];
  aux->compcode = btune->codecs->list[0];
  if (btune->config.comp_mode == BTUNE_COMP_HCR) {
    best->clevel = 8;
    aux->clevel = 8;
  }
  best->shufflesize = cctx->typesize;  // TODO typesize -> shufflesize
  aux->shufflesize = cctx->typesize;  // TODO typesize -> shufflesize
  best->nthreads_comp = cctx->nthreads;
  aux->nthreads_comp = cctx->nthreads;
  if (dctx != NULL){
    btune->max_threads = (cctx->nthreads > dctx->nthreads) ? cctx->nthreads: dctx->nthreads;
    best->nthreads_decomp = dctx->nthreads;
    aux->nthreads_decomp = dctx->nthreads;
    btune->nthreads_decomp = dctx->nthreads;
  } else {
    btune->max_threads = cctx->nthreads;
    best->nthreads_decomp = cctx->nthreads;
    aux->nthreads_decomp = cctx->nthreads;
    btune->nthreads_decomp = cctx->nthreads;
  }

  // Aux arrays to calculate the mean
  btune->current_cratios = malloc(sizeof(double)) ;
  btune->current_scores = malloc(sizeof(double));

  if (btune->config.perf_mode == BTUNE_PERF_DECOMP) {
    btune->threads_for_comp = false;
  } else {
    btune->threads_for_comp = true;
  }

  // cparams_hint
  if (config->cparams_hint) {
    extract_btune_cparams(cctx, btune->best);
    extract_btune_cparams(cctx, btune->aux_cparams);
    add_codec(btune->codecs, cctx->compcode);
    if (btune->config.behaviour.nhards_before_stop > 0) {
      if (btune->config.behaviour.nsofts_before_hard > 0){
        init_soft(btune);
      } else if (btune->config.behaviour.nwaits_before_readapt > 0) {
        btune->state = WAITING;
        btune->readapt_from = WAIT;
      } else {
        init_hard(btune);
      }
    } else {
      init_without_hards(cctx);
    }
  } else {
    init_hard(btune);
    btune->config.behaviour.nhards_before_stop++;
  }
  if (btune->config.behaviour.nhards_before_stop == 1) {
    btune->step_size = SOFT_STEP_SIZE;
  } else {
    btune->step_size = HARD_STEP_SIZE;
  }
}

// Free btune_struct
void btune_free(blosc2_context * context) {
  btune_struct * btune = context->btune;
  free(btune->codecs->list);
  free(btune->codecs);
  free(btune->best);
  free(btune->aux_cparams);
  free(btune->current_scores);
  free(btune->current_cratios);
  free(btune);
  context->btune = NULL;
}

/* Whether a codec is meant for High Compression Ratios
   Includes LZ4 + BITSHUFFLE here, but not BloscLZ + BITSHUFFLE because,
   for some reason, the latter does not work too well */
static bool is_HCR(blosc2_context * context) {
  switch (context->compcode) {
    case BLOSC_BLOSCLZ :
      return false;
    case BLOSC_LZ4 :
      return (context->filter_flags & BLOSC_DOBITSHUFFLE) ? true : false;
    case BLOSC_LZ4HC :
      return true;
    case BLOSC_ZLIB :
      return true;
    case BLOSC_ZSTD :
      return true;
    default :
      fprintf(stderr, "Error in is_COMP_HCR: codec %d not handled\n",
              context->compcode);
  }
  return false;
}

// Set the automatic blocksize 0 to its real value
void btune_next_blocksize(blosc2_context *context) {
  if (BTUNE_DISABLE_BLOCKSIZE) {
    return;
  }
  int32_t clevel = context->clevel;
  int32_t typesize = context->typesize;
  size_t nbytes = context->sourcesize;
  int32_t user_blocksize = context->blocksize;
  int32_t blocksize = (int32_t) nbytes;

  // Protection against very small buffers
  if (nbytes < typesize) {
    context->blocksize = 1;
    return;
  }

  if (user_blocksize) {
    blocksize = user_blocksize;
    // Check that forced blocksize is not too small
    if (blocksize < BLOSC_MIN_BUFFERSIZE) {
      blocksize = BLOSC_MIN_BUFFERSIZE;
    }
  }
  else if (nbytes >= L1) {
    blocksize = L1;

    /* For HCR codecs, increase the block sizes by a factor of 2 because they
        are meant for compressing large blocks (i.e. they show a big overhead
        when compressing small ones). */
    if (is_HCR(context)) {
      blocksize *= 2;
    }

    // Choose a different blocksize depending on the compression level
    switch (clevel) {
      case 0:
        // Case of plain copy
        blocksize /= 4;
        break;
      case 1:
        blocksize /= 2;
        break;
      case 2:
        blocksize *= 1;
        break;
      case 3:
        blocksize *= 2;
        break;
      case 4:
      case 5:
        blocksize *= 4;
        break;
      case 6:
      case 7:
      case 8:
        blocksize *= 8;
        break;
      case 9:
        // Do not exceed 256 KB for non HCR codecs
        blocksize *= 8;
        if (is_HCR(context)) {
          blocksize *= 2;
        }
        break;
      default:
        break;
    }
  }

  /* Enlarge the blocksize */
  if (clevel > 0) {
    if (blocksize > (1 << 16)) {
      /* Do not use a too large buffer (64 KB) for splitting codecs */
      blocksize = (1 << 16);
    }
    blocksize *= typesize;
    if (blocksize < (1 << 16)) {
      /* Do not use a too small blocksize (< 64 KB) when typesize is small */
      blocksize = (1 << 16);
    }
  }

  /* Check that blocksize is not too large */
  if (blocksize > (int32_t)nbytes) {
    blocksize = (int32_t)nbytes;
  }

  // blocksize *must absolutely* be a multiple of the typesize
  if (blocksize > typesize) {
    blocksize = (int32_t) (blocksize / typesize * typesize);
  }

  context->blocksize = blocksize;
}

// Set the cparams_btune inside blosc2_context
static void set_btune_cparams(blosc2_context * context, cparams_btune * cparams){
  context->compcode = cparams->compcode;
  context->filters[BLOSC2_MAX_FILTERS - 1] = cparams->filter;
  context->splitmode = cparams->splitmode;
  context->clevel = cparams->clevel;
  btune_struct * btune = (btune_struct*) context->btune;
  // Do not set a too large clevel for ZSTD and BALANCED mode
  if (btune->config.comp_mode == BTUNE_COMP_BALANCED &&
      (cparams->compcode == BLOSC_ZSTD || cparams->compcode == BLOSC_ZLIB) &&
      cparams->clevel >= 3) {
    cparams->clevel = 3;
  }
  // Do not set a too large clevel for HCR mode
  if (btune->config.comp_mode == BTUNE_COMP_HCR && cparams->clevel >= 6) {
    cparams->clevel = 6;
  }
  if (cparams->blocksize) {
    context->blocksize = cparams->blocksize;
  } else {
    btune_next_blocksize(context);
    cparams->blocksize = context->blocksize;
  }
  context->typesize = cparams->shufflesize;  // TODO typesize -> shufflesize
  context->new_nthreads = (int16_t) cparams->nthreads_comp;
  if (btune->dctx != NULL) {
    btune->dctx->new_nthreads = (int16_t) cparams->nthreads_decomp;
  } else {
  btune->nthreads_decomp = cparams->nthreads_decomp;
  }
}

// Tune some compression parameters based on the context
void btune_next_cparams(blosc2_context *context) {
  btune_struct * btune = (btune_struct*) context->btune;

  // Run inference only for the first chunk
  int compcode;
  uint8_t filter;
  int nchunk = context->schunk->nchunks;
  if (nchunk == 0) {
    int error = btune_model_inference(context, &compcode, &filter);
    if (error == 0) {
      printf("*** Inference Chunk #%d codec=%d filter=%d\n", nchunk, compcode, filter);
      btune->codecs->size = 1;
      btune->codecs->list[0] = compcode;
    }
  }

  *btune->aux_cparams = *btune->best;
  cparams_btune * cparams = btune->aux_cparams;

  switch(btune->state){

    // Tune codec and filter
    case CODEC_FILTER:
      int codec_index = btune->aux_index / btune->filter_split_limit;
      int compcode = btune->codecs->list[codec_index];
      int filter_split = btune->filter_split_limit;
      // Cycle filters every time
      uint8_t filter = (uint8_t) (btune->aux_index % (filter_split / 2));
      // Cycle split every two filters
      int splitmode = ((btune->aux_index % filter_split) / 3) + 1;
      if (compcode == BLOSC_BLOSCLZ) {
          // BLOSCLZ is not designed to compress well in non-split mode, so disable it always
          splitmode = BLOSC_ALWAYS_SPLIT;
      }
      // The first tuning of ZSTD in some modes should start in clevel 3
      if (((btune->config.perf_mode == BTUNE_PERF_COMP) ||
           (btune->config.perf_mode == BTUNE_PERF_BALANCED)) &&
          (compcode == BLOSC_ZSTD || cparams->compcode == BLOSC_ZLIB) &&
          (btune->nhards == 0)) {
        cparams->clevel = 3;
      }
      cparams->compcode = compcode;
      cparams->filter = filter;
      cparams->splitmode = splitmode;
      // Force auto blocksize
      // cparams->blocksize = 0;
      btune->aux_index++;
      break;

    // Tune shuffle size
    case SHUFFLE_SIZE:
      btune->aux_index++;
      if (cparams->increasing_shuffle) {
        // TODO These kind of condition checks should be removed (maybe asserts)
        if (cparams->shufflesize < MAX_SHUFFLE) {
          cparams->shufflesize <<= 1;
        }
      } else {
        int min_shuffle = (cparams->filter == 1) ? MIN_SHUFFLE: MIN_BITSHUFFLE;
        if (cparams->shufflesize > min_shuffle) {
          cparams->shufflesize >>= 1;
        }
      }
      break;

    // Tune the number of threads
    case THREADS:
      btune->aux_index++;
      int * nthreads;
      if (btune->threads_for_comp) {
        nthreads = &cparams->nthreads_comp;
      } else {
        nthreads = &cparams->nthreads_decomp;
      }
      if (cparams->increasing_nthreads) {
        if (*nthreads < btune->max_threads) {
          (*nthreads)++;
        }
      } else {
        if (*nthreads > MIN_THREADS) {
          (*nthreads)--;
        }
      }
      break;

    // Tune compression level
    case CLEVEL:
      // Force auto blocksize on hard readapts
      if (btune->readapt_from == HARD){
        cparams->blocksize = 0;
      }
      btune->aux_index++;
      if (cparams->increasing_clevel) {
        if (cparams->clevel <= (MAX_CLEVEL - btune->step_size)) {
          cparams->clevel += btune->step_size;
          // ZSTD level 9 is extremely slow, so avoid it, always
          if (cparams->clevel == 9 && cparams->compcode == BLOSC_ZSTD) {
            cparams->clevel = 8;
          }
        }
      } else {
        if (cparams->clevel > btune->step_size) {
          cparams->clevel -= btune->step_size;
        }
      }
      break;

    // Tune block size
    case BLOCKSIZE:
      btune->aux_index++;
      if (BTUNE_DISABLE_BLOCKSIZE) {
        break;
      }
      int step_factor = btune->step_size - 1;
      if (cparams->increasing_block) {
        int32_t new_block = cparams->blocksize * 1 << btune->step_size;
        if ((cparams->blocksize <= (MAX_BLOCK >> step_factor)) &&
            (new_block <= context->sourcesize)) {
          cparams->blocksize = new_block;
        }
      } else {
        if (cparams->blocksize >= (MIN_BLOCK << step_factor)) {
          cparams->blocksize >>= btune->step_size;
        }
      }
      break;

    // Try without compressing
    case MEMCPY:
      btune->aux_index++;
      cparams->clevel = 0;
      break;

    // Waiting
    case WAITING:
      btune->nwaitings++;
      break;

    // Stopped
    case STOP:
      return;
  }
  set_btune_cparams(context, cparams);
}

// Computes the score depending on the perf_mode
static double score_function(btune_struct * btune, double ctime, size_t cbytes,
                             double dtime) {
  double reduced_cbytes = (double)cbytes / (double) BTUNE_KB;
  switch (btune->config.perf_mode) {
    case BTUNE_PERF_COMP:
      return ctime + reduced_cbytes / btune->config.bandwidth;
    case BTUNE_PERF_DECOMP:
      return reduced_cbytes / btune->config.bandwidth + dtime;
    case BTUNE_PERF_BALANCED:
      return ctime + reduced_cbytes / btune->config.bandwidth + dtime;
    default:
      fprintf(stderr, "WARNING: unknown performance mode\n");
      return -1;
  }
}

static double mean(double const * array, int size) {
  double sum = 0;
  for (int i = 0; i < size; i++) {
    sum += array[i];
  }
  return sum / size;
}

// Determines if btune has improved depending on the comp_mode
static bool has_improved(btune_struct * btune, double score_coef, double cratio_coef) {
  btune_comp_mode comp_mode = btune->config.comp_mode;
  switch (comp_mode) {
    case BTUNE_COMP_HSP:
      return (((cratio_coef > 1) && (score_coef > 1)) ||
              ((cratio_coef > 0.5) && (score_coef > 2)) ||
              ((cratio_coef > 0.67) && (score_coef > 1.3)) ||
              ((cratio_coef > 2) && (score_coef > 0.7)));
    case BTUNE_COMP_BALANCED:
      return (((cratio_coef > 1) && (score_coef > 1)) ||
              ((cratio_coef > 1.1) && (score_coef > 0.8)) ||
              ((cratio_coef > 1.3) && (score_coef > 0.5)));
    case BTUNE_COMP_HCR:
      return cratio_coef > 1;
    default:
      fprintf(stderr, "WARNING: unknown compression mode\n");
      return false;

  }
}


static bool cparams_equals(cparams_btune * cp1, cparams_btune * cp2) {
  return ((cp1->compcode == cp2->compcode) &&
          (cp1->filter == cp2->filter) &&
          (cp1->splitmode == cp2->splitmode) &&
          (cp1->clevel == cp2->clevel) &&
          (cp1->blocksize == cp2->blocksize) &&
          (cp1->shufflesize == cp2->shufflesize) &&
          (cp1->nthreads_comp == cp2->nthreads_comp) &&
          (cp1->nthreads_decomp == cp2->nthreads_decomp));
}


// Processes which btune_state will come next after a readapt or wait
static void process_waiting_state(blosc2_context * ctx) {
  btune_struct * btune = (btune_struct*) ctx->btune;
  btune_behaviour behaviour = btune->config.behaviour;
  uint32_t minimum_hards = 0;

  if (!btune->config.cparams_hint) {
    minimum_hards++;
  }

  char* envvar = getenv("BTUNE_LOG");
  if (envvar != NULL) {
    // Print the winner of the readapt
//  if (btune->readapt_from != WAIT && !btune->is_repeating) {
//    char* compname;
//    blosc_compcode_to_compname(cparams->compcode, &compname);
//    printf("| %10s | %d | %d | %d | %d | %d | %d | %.3g | %.3gx | %s\n",
//           compname, cparams->filter, cparams->clevel,
//           (int) cparams->blocksize / BTUNE_KB, (int) cparams->shufflesize,
//           cparams->nthreads_comp, cparams->nthreads_decomp,
//           cparams->score, cparams->cratio, "WINNER");
//  }
  }

  switch (btune->readapt_from) {
    case HARD:
      btune->nhards++;
      assert(btune->nhards > 0);
      // Last hard (initial readapts completed)
      if ((behaviour.nhards_before_stop == minimum_hards) ||
          (btune->nhards % behaviour.nhards_before_stop == 0)) {
        btune->is_repeating = true;
        // There are softs (repeat_mode no stop)
        if ((behaviour.nsofts_before_hard > 0) &&
            (behaviour.repeat_mode != BTUNE_STOP)) {
          init_soft(btune);
        // No softs (repeat_mode soft)
        } else if (behaviour.repeat_mode != BTUNE_REPEAT_ALL) {
          btune->state = STOP;
        // No softs, there are waits (repeat_mode all)
        } else if (behaviour.nwaits_before_readapt > 0) {
          btune->state = WAITING;
          btune->readapt_from = WAIT;
        // No softs, no waits and there are hards (repeat_mode all)
        } else if (behaviour.nhards_before_stop > minimum_hards) {
          init_hard(btune);
        // No softs, no waits no hards (repeat_mode all)
        } else {
          btune->state = STOP;
        }
        // Not the last hard (there are softs readapts)
      } else if (behaviour.nsofts_before_hard > 0) {
        init_soft(btune);
        // No softs but there are waits
      } else if (behaviour.nwaits_before_readapt > 0) {
        btune->state = WAITING;
        btune->readapt_from = WAIT;
        // No softs, no waits
      } else {
        init_hard(btune);
      }
      break;

    case SOFT:
      btune->nsofts++;
      btune->readapt_from = WAIT;
      assert(btune->nsofts > 0);
      if (behaviour.nwaits_before_readapt == 0) {
        // Last soft
        if (((behaviour.nsofts_before_hard == 0) ||
            (btune->nsofts % behaviour.nsofts_before_hard == 0)) &&
            !(btune->is_repeating && (behaviour.repeat_mode != BTUNE_REPEAT_ALL)) &&
            (behaviour.nhards_before_stop > minimum_hards)) {
          init_hard(btune);
        // Special, hint true, no hards, last soft, stop_mode
        } else if ((minimum_hards == 0) &&
                   (behaviour.nhards_before_stop == 0) &&
                   (btune->nsofts % behaviour.nsofts_before_hard == 0) &&
                   (behaviour.repeat_mode == BTUNE_STOP)) {
          btune->is_repeating = true;
          btune->state = STOP;
        // Not the last soft
        } else {
          init_soft(btune);
        }
      }
      break;

    case WAIT:
      // Last wait
      if ((behaviour.nwaits_before_readapt == 0) ||
          ((btune->nwaitings != 0) &&
          (btune->nwaitings % behaviour.nwaits_before_readapt == 0))) {
        // Last soft
        if (((behaviour.nsofts_before_hard == 0) ||
            ((btune->nsofts != 0) &&
            (btune->nsofts % behaviour.nsofts_before_hard == 0))) &&
            !(btune->is_repeating && (behaviour.repeat_mode != BTUNE_REPEAT_ALL)) &&
            (behaviour.nhards_before_stop > minimum_hards)) {

          init_hard(btune);
        // Not last soft
        } else if ((behaviour.nsofts_before_hard > 0) &&
                   !(btune->is_repeating && (behaviour.repeat_mode == BTUNE_STOP))){

          init_soft(btune);
        }
      }
  }
  // Force soft step size on last hard
  if ((btune->readapt_from == HARD) &&
      (btune->nhards == (int)(behaviour.nhards_before_stop - 1))) {
    btune->step_size = SOFT_STEP_SIZE;
  }
}

// State transition handling
static void update_aux(blosc2_context * ctx, bool improved) {
  btune_struct * btune = ctx->btune;
  cparams_btune * best = btune->best;
  bool first_time = btune->aux_index == 1;
  switch (btune->state) {
    case CODEC_FILTER:
      // Reached last combination of codec filter
      if ((btune->aux_index / btune->filter_split_limit) == btune->codecs->size) {
        btune->aux_index = 0;

        int32_t shufflesize = best->shufflesize;
        // Is shufflesize valid or not
        if (BTUNE_DISABLE_SHUFFLESIZE) {
          if (!BTUNE_DISABLE_THREADS) {
            btune->state = THREADS;
          }
          else {
            btune->state = CLEVEL;
          }
        } else {
          bool is_power_2 = (shufflesize & (shufflesize - 1)) == 0;
          btune->state = (best->filter && is_power_2) ? SHUFFLE_SIZE : THREADS;
        }
        // max_threads must be greater than 1
        if ((btune->state == THREADS) && (btune->max_threads == 1)) {
          btune->state = CLEVEL;
          if (has_ended_clevel(btune)) {
            best->increasing_clevel = !best->increasing_clevel;
          }
        }
        // Control direction parameters
        if (!BTUNE_DISABLE_SHUFFLESIZE && btune->state == SHUFFLE_SIZE) {
          if (has_ended_shuffle(best)) {
            best->increasing_shuffle = !best->increasing_shuffle;
          }
        } else if (btune->state == THREADS) {
          if (has_ended_shuffle(best)) {
            best->increasing_nthreads = !best->increasing_nthreads;
          }
        }
      }
      break;

    case SHUFFLE_SIZE:
      if (!improved && first_time) {
        best->increasing_shuffle = !best->increasing_shuffle;
      }
      // Can not change parameter or is not improving
      if (has_ended_shuffle(best) || (!improved && !first_time)) {
        btune->aux_index = 0;
          if (!BTUNE_DISABLE_THREADS) {
            btune->state = THREADS;
          }
          else {
            btune->state = CLEVEL;
          }
        // max_threads must be greater than 1
        if ((btune->state == THREADS) && (btune->max_threads == 1)) {
          btune->state = CLEVEL;
          if (has_ended_clevel(btune)) {
            best->increasing_clevel = !best->increasing_clevel;
          }
        } else {
          if (has_ended_threads(btune)) {
            best->increasing_nthreads = !best->increasing_nthreads;
          }
        }
      }
      break;

    case THREADS:
      first_time = (btune->aux_index % MAX_STATE_THREADS) == 1;
      if (!improved && first_time) {
        best->increasing_nthreads = !best->increasing_nthreads;
      }
      // Can not change parameter or is not improving
      if (has_ended_threads(btune) || (!improved && !first_time)) {
        // If perf_mode BALANCED mark btune to change threads for decompression
        if (btune->config.perf_mode == BTUNE_PERF_BALANCED) {
          if (btune->aux_index < MAX_STATE_THREADS) {
            btune->threads_for_comp = !btune->threads_for_comp;
            btune->aux_index = MAX_STATE_THREADS;
            if (has_ended_threads(btune)) {
              best->increasing_nthreads = !best->increasing_nthreads;
            }
          }
        // No BALANCED mark to end
        } else {
          btune->aux_index = MAX_STATE_THREADS + 1;
        }
        // THREADS ended
        if (btune->aux_index > MAX_STATE_THREADS) {
          btune->aux_index = 0;
          btune->state = CLEVEL;
          if (has_ended_clevel(btune)) {
            best->increasing_clevel = !best->increasing_clevel;
          }
        }
      }
      break;

    case CLEVEL:
      if (!improved && first_time) {
        best->increasing_clevel = !best->increasing_clevel;
      }
      // Can not change parameter or is not improving
      if (has_ended_clevel(btune) || (!improved && !first_time)) {
        btune->aux_index = 0;
        if (!BTUNE_DISABLE_BLOCKSIZE) {
          btune->state = BLOCKSIZE;
        }
        else {
          if (!BTUNE_DISABLE_MEMCPY) {
            btune->state = MEMCPY;
          }
          else {
            btune->state = WAITING;
          }
        }
        if (has_ended_blocksize(ctx)) {
          best->increasing_block = !best->increasing_block;
        }
      }
      break;

    case BLOCKSIZE:
      if (!improved && first_time) {
        best->increasing_block = !best->increasing_block;
      }
      // Can not change parameter or is not improving
      if (has_ended_blocksize(ctx) || (!improved && !first_time)) {
        btune->aux_index = 0;
        if (btune->config.comp_mode == BTUNE_COMP_HSP) {
          if (!BTUNE_DISABLE_MEMCPY) {
            btune->state = MEMCPY;
          }
          else {
            btune->state = WAITING;
          }
        } else {
          btune->state = WAITING;
        }
      }
      break;

    case MEMCPY:
      btune->aux_index = 0;
      btune->state = WAITING;
      break;

    default:
      ;
  }
  if (btune->state == WAITING) {
    process_waiting_state(ctx);
  }
}

// Update btune structs with the compression results
void btune_update(blosc2_context * context, double ctime) {
  btune_struct * btune = (btune_struct*)(context->btune);
  if (btune->state == STOP) {
    return;
  }

  btune->steps_count++;
  cparams_btune * cparams = btune->aux_cparams;

  // We come from blosc_compress_context(), so we can populate metrics now
  size_t cbytes = context->destsize;
  double dtime = 0;

  // When the source is NULL (eval with prefilters), decompression is not working.
  // Disabling this part for the time being.
//  // Compute the decompression time if needed
//  btune_behaviour behaviour = btune->config.behaviour;
//  if (!((btune->state == WAITING) &&
//      ((behaviour.nwaits_before_readapt == 0) ||
//      (btune->nwaitings % behaviour.nwaits_before_readapt != 0))) &&
//      ((btune->config.perf_mode == BTUNE_PERF_DECOMP) ||
//      (btune->config.perf_mode == BTUNE_PERF_BALANCED))) {
//    blosc2_context * dctx;
//    if (btune->dctx == NULL) {
//      blosc2_dparams params = { btune->nthreads_decomp, NULL, NULL, NULL};
//      dctx = blosc2_create_dctx(params);
//    } else {
//      dctx = btune->dctx;
//    }
//    blosc_set_timestamp(&last);
//    blosc2_decompress_ctx(dctx, context->dest, context->destsize, (void*)(context->src),
//                          context->sourcesize);
//    blosc_set_timestamp(&current);
//    dtime = blosc_elapsed_secs(last, current);
//    if (btune->dctx == NULL) {
//      blosc2_free_ctx(dctx);
//    }
//  }

  double score = score_function(btune, ctime, cbytes, dtime);
  assert(score > 0);
  double cratio = (double) context->sourcesize / (double) cbytes;

  cparams->score = score;
  cparams->cratio = cratio;
  cparams->ctime = ctime;
  cparams->dtime = dtime;
  btune->current_scores[btune->rep_index] = score;
  btune->current_cratios[btune->rep_index] = cratio;
  btune->rep_index++;
  if (btune->rep_index == 1) {
    score = mean(btune->current_scores, 1);
    cratio = mean(btune->current_cratios, 1);
    double cratio_coef = cratio / btune->best->cratio;
    double score_coef = btune->best->score / score;
    bool improved;
    // In state THREADS the improvement comes from ctime or dtime
    if (btune->state == THREADS) {
      if (btune->threads_for_comp) {
        improved = ctime < btune->best->ctime;
      } else {
        improved = dtime < btune->best->dtime;
      }
    } else {
      improved = has_improved(btune, score_coef, cratio_coef);
    }
    char winner = '-';
    // If the chunk is made of special values, it cannot never improve scoring
    if (cbytes <= (BLOSC2_MAX_OVERHEAD + (size_t)context->typesize)) {
      improved = false;
      winner = 'S';
    }
    if (improved) {
      winner = 'W';
    }

    if (!btune->is_repeating) {
      char* envvar = getenv("BTUNE_LOG");
      if (envvar != NULL) {
        int split = (cparams->splitmode == BLOSC_ALWAYS_SPLIT) ? 1 : 0;
        const char *compname;
        blosc2_compcode_to_compname(cparams->compcode, &compname);
        printf("| %10s | %6d | %5d | %7d | %9d | %11d | %9d | %9d | %9.3g | %9.3gx | %15s | %7s | %c\n",
               compname, cparams->filter, split, cparams->clevel,
               (int) cparams->blocksize / BTUNE_KB, (int) cparams->shufflesize,
               cparams->nthreads_comp, cparams->nthreads_decomp,
               score, cratio, stcode_to_stname(btune), readapt_to_str(btune->readapt_from), winner);
      }
    }

    // if (improved || cparams_equals(btune->best, cparams)) {
    // We don't want to get rid of the previous best->score
    if (improved) {
      *btune->best = *cparams;
    }
    btune->rep_index = 0;
    update_aux(context, improved);
  }
}
