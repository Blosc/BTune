/**********************************************************************
  Optimize Blosc2 parameters using deep/machine learning.

  Copyright (c) 2023 The Blosc Developers <blosc@blosc.org>
  License: GNU Affero General Public License v3.0

  See COPYING.txt for details about copyright and rights to use.
***********************************************************************/

/** @file  btune.h
 * @brief BTune header file.
 *
 * This file contains the public methods and structures needed to use BTune.
 */

#ifndef BTUNE_H
#define BTUNE_H

#include <stdbool.h>
#include "context.h"

// The size of L1 cache.  32 KB is quite common nowadays.
#define L1 (32 * 1024)
/* Version numbers */
#define BTUNE_VERSION_MAJOR    1    /* for major interface/format changes  */
#define BTUNE_VERSION_MINOR    0    /* for minor interface/format changes  */
#define BTUNE_VERSION_RELEASE  0    /* for tweaks, bug-fixes, or development */
#define BTUNE_VERSION_STRING "1.0.0"

#define BTUNE_DEBUG(msg, ...) \
    do { \
         const char *__e = getenv("BTUNE_DEBUG"); \
         if (!__e) { break; } \
         fprintf(stderr, "[DEBUG] " msg "\n", ##__VA_ARGS__); \
       } while(0)

/**
 * @brief BTune units enumeration.
 *
 * This enumeration provides the most common units of bandwidth for its use
 * in the BTune config. The bandwidth units are expressed in kB/s.
*/
enum bandwidth_units{
  BTUNE_MBPS = 1024,                                  //!< A 1 MB/s bandwidth expressed in kB/s, 1024 kB/s.
  BTUNE_MBPS10 = 10 * BTUNE_MBPS,                     //!< A 10 MB/s bandwidth expressed in kB/s, 10240 kB/s.
  BTUNE_MBPS100 = 100 * BTUNE_MBPS,                   //!< A 100 MB/s bandwidth expressed in kB/s, 102400 kB/s.
  BTUNE_GBPS = 1 * BTUNE_MBPS * BTUNE_MBPS,           //!< A 1 GB/s bandwidth expressed in kB/s, 1024^2 kB/s.
  BTUNE_GBPS10 = 10 * BTUNE_MBPS * BTUNE_MBPS,        //!< A 10 GB/s bandwidth expressed in kB/s, 10 * 1024^2 kB/s.
  BTUNE_GBPS100 = 100 * BTUNE_MBPS * BTUNE_MBPS,      //!< A 100 GB/s bandwidth expressed in kB/s, 100 * 1024^2 kB/s.
  BTUNE_TBPS = BTUNE_MBPS * BTUNE_MBPS * BTUNE_MBPS,  //!< A 1 TB/s bandwidth expressed in kB/s, 1024^3 kB/s.
};

/**
 * @brief Compression mode enumeration.
 *
 * The compression mode alters the BTune criteria for improvement.
 * Depending on this value BTune will prioritize the compression speed,
 * the compression ratio or both.
*/
typedef enum {
  BTUNE_COMP_HSP,       //!< Optimizes the speed, even accepting memcpy.
  BTUNE_COMP_BALANCED,  //!< Optimizes both, the speed and compression ratio.
  BTUNE_COMP_HCR,       //!< Optimizes the compression ratio.
} btune_comp_mode;

/**
 * @brief Performance mode enumeration.
 *
 * The performance mode alters the BTune scoring function used for improvement.
 * Depending on this value BTune will consider for the scoring either the compression time or
 * the decompression time, or both.
*/
typedef enum {
  BTUNE_PERF_COMP,     //!< Optimizes the compression and transmission times.
  BTUNE_PERF_DECOMP,   //!< Optimizes the decompression and transmission times.
  BTUNE_PERF_BALANCED  //!< Optimizes the compression, transmission and decompression times.
} btune_performance_mode;

/**
 * @brief Repeat mode enumeration.
 *
 * Changes the way BTune behaves when it has completed all the initial readaptations.
 * @see #btune_behaviour
*/
typedef enum {
  BTUNE_STOP,         //!< BTune will stop improving.
  BTUNE_REPEAT_SOFT,  //!< BTune will repeat only the soft readapts continuously.
  BTUNE_REPEAT_ALL,   //!< BTune will repeat the initial readaptations continuously.
} btune_repeat_mode;

/**
 * @brief BTune behaviour struct.
 *
 * This specifies the number of initial hard readapts,
 * the number of soft readapts between each hard readapt and the number of waits,
 * before initiating a readapt.
 * Note: a readapt is the process by which btune adjusts the compression parameters.
 * It can be of two types: \b soft, which only changes the compression level and
 * blocksize or \b hard, which also changes the codec, filters, shuffle size and number of threads.
*/
typedef struct {
  uint32_t nwaits_before_readapt;
  /**< Number of waiting states before a readapt.
   *
   * During a waiting state BTune will not alter any compression parameter.
  */
  uint32_t nsofts_before_hard;
  //!< Number of soft readapts before a hard readapt.
  uint32_t nhards_before_stop;
  //!< Number of initial hard readapts.
  btune_repeat_mode repeat_mode;
  /**< BTune repeat mode.
   *
   * Once completed the initial hard readapts, the repeat mode will determine
   * if BTune continues repeating readapts or stops permanently.
  */
} btune_behaviour;

/**
 * @brief BTune configuration struct.
 *
 * The btune_config struct contains all the parameters used by BTune which determine
 * how the compression parameters will be tuned.
*/
typedef struct {
  uint32_t bandwidth;
  /**< The bandwidth to which optimize in kB/s.
   *
   * Used to calculate the transmission times.
  */
  btune_performance_mode perf_mode;
  //!< The BTune performance mode.
  btune_comp_mode comp_mode;
  //!< The BTune compression mode.
  btune_behaviour behaviour;
  //!< The BTune behaviour config.
  bool cparams_hint;
  /**< Whether use the cparams specified in the context or not.
   *
   * When true, this will force BTune to use the cparams provided inside the context, note
   * that after a hard readapt the cparams will change.
   * When false, BTune will start from a hard readapt to determine the best cparams, note
   * that this hard readapt is not considered for the number of initial hard readapts.
   * @see #btune_behaviour
  */

} btune_config;

/**
 * @brief BTune default configuration.
 *
 * This default configuration of BTune is meant for optimizing memory bandwidth, compression speed,
 * decompression speed and the compression ratio (BALANCED options). It behaves as follows:
 * it starts with a hard readapt (cparams_hint false) and then repeats 5 soft readapts and
 * a hard readapt 1 times before stopping completely.
*/
static const btune_config BTUNE_CONFIG_DEFAULTS = {
    2 * BTUNE_GBPS10,
    BTUNE_PERF_BALANCED,
    BTUNE_COMP_BALANCED,
    {0, 5, 1, BTUNE_STOP},
    false
};

/// @cond DEV
// Internal BTune state enumeration.
typedef enum {
    CODEC_FILTER,
    SHUFFLE_SIZE,
    THREADS,
    CLEVEL,
    BLOCKSIZE,
    MEMCPY,
    WAITING,
    STOP,
} btune_state;

// Internal BTune readapt type
typedef enum {
    WAIT,
    SOFT,
    HARD,
} readapt_type;

// Internal BTune codec list
typedef struct {
    int * list;
    int size;
} codec_list;

// Internal BTune compression parameters
typedef struct {
    int compcode;
    // The compressor code
    uint8_t filter;
    // The precompression filter
    int32_t splitmode;
    // Whether the blocks should be split or not
    int clevel;
    // The compression level
    int32_t blocksize;
    // The block size
    int32_t shufflesize;
    // The shuffle size
    int nthreads_comp;
    // The number of threads used for compressing
    int nthreads_decomp;
    // The number of threads used for decompressing
    bool increasing_clevel;
    // Control parameter for clevel
    bool increasing_block;
    // Control parameter for blocksize
    bool increasing_shuffle;
    // Control parameter for shufflesize
    bool increasing_nthreads;
    // Control parameter for nthreads
    double score;
    // The score obtained with this cparams
    double cratio;
    // The cratio obtained with this cparams
    double ctime;
    // The compression time obtained with this cparams
    double dtime;
    // The decompression time obtained with this cparams
} cparams_btune;

// BTune struct
typedef struct {
  btune_config config;
  // The BTune config
  codec_list * codecs;
  // The codec list used by BTune
  cparams_btune * best;
  // The best cparams optained with BTune
  cparams_btune * aux_cparams;
  // The aux cparams for updating the best
  double * current_scores;
  // The aux array of scores to calculate the mean
  double * current_cratios;
  // The aux array of cratios to calculate the mean
  int rep_index;
  // The aux index for the repetitions
  int aux_index;
  // The auxiliar index for state management
  int steps_count;
  // The count of steps made made by BTune
  btune_state state;
  // The state of BTune
  int step_size;
  // The step size within clevels and blocksizes
  int nwaitings;
  // The total count of nwaitings states
  int nsofts;
  // The total count of soft readapts
  int nhards;
  // The total count of hard readapts
  bool is_repeating;
  // If BTune has completed the initial readapts
  readapt_type readapt_from;
  // If BTune is making a hard or soft readapt, or is WAITING
  int max_threads;
  // The maximum number of threads used
  blosc2_context * dctx;
  // The decompression context
  int nthreads_decomp;
  // The number of threads for decompression (used if dctx is NULL)
  bool threads_for_comp;
  // Depending on this value the THREADS state will change the compression or decompression threads
} btune_struct;
/// @endcond

/**
 * @brief BTune initializer.
 *
 * This method initializes BTune in the compression context and then it will be used automatically.
 * On each compression, BTune overwrites the compression parameters in the context and, depending
 * on the results obtained and its configuration, will adjust them.
 * Example of use:
 * @code{.c}
 * blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
 * blosc2_dparams params = BLOSC2_DPARAMS_DEFAULTS;
 * btune_config config = BTUNE_CONFIG_DEFAULTS;
 * blosc2_storage storage = {.cparams=&cparams, .dparams=&dparams};
 * blosc2_schunk * schunk = blosc2_schunk_new(&storage);
 * btune_init(&config, schunk->cctx, schunk->dctx);
 * @endcode
 * @param config The BTune configuration determines its behaviour and how will optimize.
 * @param cctx The compression context where BTune tunes the compression parameters. It <b>can not</b> be NULL.
 * @param dctx If not NULL, BTune will modify the number of threads for decompression inside this context.
*/
void btune_init(void * config, blosc2_context* cctx, blosc2_context* dctx);

void btune_free(blosc2_context* context);

void btune_next_cparams(blosc2_context *context);

void btune_update(blosc2_context* context, double ctime);

void btune_next_blocksize(blosc2_context *context);

#endif  /* BTUNE_H */
