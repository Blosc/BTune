/*
  Copyright (C) 2021  The Blosc Developers <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  Example program demonstrating use of the Blosc entropy probe codec from C code.

  To compile this program:

  $ gcc entropy_probe.c -o entropy_probe -lblosc2

  To run:

  $ ./entropy_probe <filename.b2frame>

  To run in entropy mode:

  $ ./entropy_probe -e <filename.b2frame>
*/

#include <stdio.h>
#include "blosc2.h"
#include "blosc2_entropy_prober.h"
#include "blosc2/filters-registry.h"

#define KB 1024.
#define MB (1024 * KB)
#define GB (1024 * MB)

static const char *get_compname(int compcode)
{
    const char *compname;
    switch (compcode) {
        case BLOSC_BLOSCLZ:
            compname = BLOSC_BLOSCLZ_COMPNAME;
            break;
        case BLOSC_LZ4:
            compname = BLOSC_LZ4_COMPNAME;
            break;
        case BLOSC_LZ4HC:
            compname = BLOSC_LZ4HC_COMPNAME;
            break;
        case BLOSC_ZLIB:
            compname = BLOSC_ZLIB_COMPNAME;
            break;
        case BLOSC_ZSTD:
            compname = BLOSC_ZSTD_COMPNAME;
            break;
        case ENTROPY_PROBE_ID:
            compname = "entropy";
            break;
        default:
            printf("Unsupported codec!");
            exit(1);
    }

    return compname;
}

static const char *get_filtername(int nfilter)
{
    const char *sfilter;
    switch (nfilter) {
        case BLOSC_NOFILTER:
            sfilter = "nofilter";
            break;
        case BLOSC_SHUFFLE:
            sfilter = "shuffle";
            break;
        case BLOSC_BITSHUFFLE:
            sfilter = "bitshuffle";
            break;
        case BLOSC_FILTER_BYTEDELTA:
            sfilter = "shuffle-bytedelta";
            break;
        default:
            printf("Unsupported filter!");
            exit(1);
    }

    return sfilter;
}

static const char *get_splitname(int splitmode)
{
    const char *ssplit;
    switch (splitmode) {
        case BLOSC_ALWAYS_SPLIT:
            ssplit = "split";
            break;
        case BLOSC_NEVER_SPLIT:
            ssplit = "nosplit";
            break;
        default:
            printf("Unsupported splitmode!");
            exit(1);
    }

    return ssplit;
}

// This function receives an instrumented chunk having nstreams
static int extr_data(
    FILE *csv_file,
    blosc2_schunk *schunk,
    int nchunk,
    blosc2_cparams *cparams,
    blosc2_dparams *dparams,
    int category
)
{
    uint8_t chunk[schunk->chunksize];
    uint8_t chunk2[schunk->chunksize];

    printf("decompressing chunk # %d (out of %ld)\n", nchunk, schunk->nchunks);

    int dsize = blosc2_schunk_decompress_chunk(schunk, nchunk, chunk, schunk->chunksize);
    if (dsize < 0) {
        printf("Error decompressing chunk in schunk.  Error code: %d\n", dsize);
        return dsize;
    }

    blosc2_context *cctx = blosc2_create_cctx(*cparams);
    int csize = blosc2_compress_ctx(cctx, chunk, dsize, chunk2, schunk->chunksize);
    if (csize < 0) {
        printf("Error compressing chunk.  Error code: %d\n", csize);
        return csize;
    }

    blosc2_context *dctx = blosc2_create_dctx(*dparams);
    int dsize2 = blosc2_decompress_ctx(dctx, chunk2, csize, chunk, dsize);
    if (dsize2 < 0) {
        printf("Error decompressing chunk.  Error code: %d\n", dsize2);
        return dsize2;
    }

    int nstreams = dsize2 / (int)sizeof(blosc2_instr);
    printf("Chunk %d data with %d streams:\n", nchunk, nstreams);

    blosc2_instr *instr_data = (blosc2_instr *)chunk;

    for (int nstream = 0; nstream < nstreams; nstream++) {
        float cratio = instr_data->cratio;
        float cspeed = instr_data->cspeed;
        bool special_val = instr_data->flags[0];
        if (!special_val) {
            // Fill csv file
            int special_vals = 0;
            fprintf(csv_file, "%.3g, %.3g, %d, %d, %d\n", cratio, cspeed, special_vals, nchunk, category);
            printf("Chunk %d, block %d: cratio %.3g, speed %.3g\n", nchunk, nstream, cratio, cspeed);
        }
        else {
            // Fill csv file
            int special_vals = 1;
            fprintf(csv_file, "%.3g, %.3g, %d, %d, %d\n", cratio, cspeed, special_vals, nchunk, category);
            printf("Chunk %d, block %d: cratio %.3g, speed %.3g\n", nchunk, nstream, cratio, cspeed);
        }
        instr_data++;
    }

    return 0;
}

static void print_compress_info(void)
{
    char *name = NULL, *version = NULL;
    int ret;

    printf("Blosc version: %s (%s)\n", BLOSC2_VERSION_STRING, BLOSC2_VERSION_DATE);

    printf("List of supported compressors in this build: %s\n",
           blosc2_list_compressors());

    printf("Supported compression libraries:\n");
    ret = blosc2_get_complib_info("blosclz", &name, &version);
    if (ret >= 0)
        printf("  %s: %s\n", name, version);
    free(name);
    free(version);
    ret = blosc2_get_complib_info("lz4", &name, &version);
    if (ret >= 0)
        printf("  %s: %s\n", name, version);
    free(name);
    free(version);
    ret = blosc2_get_complib_info("zlib", &name, &version);
    if (ret >= 0)
        printf("  %s: %s\n", name, version);
    free(name);
    free(version);
    ret = blosc2_get_complib_info("zstd", &name, &version);
    if (ret >= 0)
        printf("  %s: %s\n", name, version);
    free(name);
    free(version);
}

int main(int argc, char *argv[])
{
    char usage[256];
    char csv_filename[256];
    char data_filename[256];
    bool entropy_probe_mode = false;

    print_compress_info();

    strcpy(usage, "Usage: entropy_probe [-e] data_filename");

    if (argc < 2) {
        printf("%s\n", usage);
        exit(1);
    }

    if (argc >= 3) {
        if (strcmp("-e", argv[1]) != 0) {
            printf("%s\n", usage);
            exit(1);
        }
        strcpy(data_filename, argv[2]);
        entropy_probe_mode = true;
    }
    else {
        strcpy(data_filename, argv[1]);
    }
    printf("fitxer %s\n", data_filename);

    blosc2_init();
    blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
    cparams.instr_codec = true;

    // The entropy probe detector is meant to always be used in SPLIT mode but
    // without split works better
    cparams.splitmode = entropy_probe_mode ? BLOSC_NEVER_SPLIT : BLOSC_ALWAYS_SPLIT;

    if (entropy_probe_mode) {
        blosc2_codec codec;
        b2ep_register_codec(&codec);
        cparams.compcode = ENTROPY_PROBE_ID;
    }

    blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;

    blosc2_schunk *schunk = blosc2_schunk_open(data_filename);
    if (schunk == NULL) {
        printf("Cannot open the data file\n");
        exit(1);
    }
    printf("nchunks in dataset: %lld\n", schunk->nchunks);
    cparams.blocksize = schunk->blocksize;
    cparams.typesize = schunk->typesize;

    if (!entropy_probe_mode) {
        // Loop over different filters and codecs
        int v_codecs[] = {0, 1, 2, 4, 5};
        int v_splits[] = {1, 2};

        int n_codecs = sizeof(v_codecs) / sizeof(int);
        for (int ncodec = 0; ncodec < n_codecs; ++ncodec) {
            for (int nfilter = 0; nfilter <= FILTER_STOP; ++nfilter) {
                for (int nsplit = 0; nsplit < BLOSC_NEVER_SPLIT; ++nsplit) {

                    cparams.splitmode = v_splits[nsplit];
                    cparams.compcode = v_codecs[ncodec];
                    if (nfilter == FILTER_STOP) {
                        nfilter = BLOSC_FILTER_BYTEDELTA;
                        cparams.filters_meta[BLOSC2_MAX_FILTERS - 1] = schunk->typesize;
                        cparams.filters[BLOSC2_MAX_FILTERS - 2] = BLOSC_SHUFFLE;
                    }
                    cparams.filters[BLOSC2_MAX_FILTERS - 1] = nfilter;

                    const char *compname = get_compname(cparams.compcode);
                    const char *sfilter = get_filtername(nfilter);
                    const char *ssplit = get_splitname(cparams.splitmode);

                    // Create csv file
                    sprintf(csv_filename, "%s-%s-%s.csv", compname, sfilter, ssplit);
                    printf("CSV filename: %s\n", csv_filename);
                    FILE *csv_file = fopen(csv_filename, "w");
                    if (csv_file == NULL) {
                        printf("Error creating the file\n");
                        return -1;
                    }

                    int category;
                    if (cparams.splitmode == BLOSC_ALWAYS_SPLIT) {
                        if (cparams.compcode == BLOSC_BLOSCLZ && nfilter == BLOSC_NOFILTER) {
                            category = 0;
                        }
                        else if (cparams.compcode == BLOSC_BLOSCLZ && nfilter == BLOSC_SHUFFLE) {
                            category = 1;
                        }
                        else if (cparams.compcode == BLOSC_BLOSCLZ && nfilter == BLOSC_BITSHUFFLE) {
                            category = 2;
                        }
                        else if (cparams.compcode == BLOSC_BLOSCLZ && nfilter == BLOSC_FILTER_BYTEDELTA) {
                            category = 3;
                        }
                        else if (cparams.compcode == BLOSC_LZ4 && nfilter == BLOSC_NOFILTER) {
                            category = 4;
                        }
                        else if (cparams.compcode == BLOSC_LZ4 && nfilter == BLOSC_SHUFFLE) {
                            category = 5;
                        }
                        else if (cparams.compcode == BLOSC_LZ4 && nfilter == BLOSC_BITSHUFFLE) {
                            category = 6;
                        }
                        else if (cparams.compcode == BLOSC_LZ4 && nfilter == BLOSC_FILTER_BYTEDELTA) {
                            category = 7;
                        }
                        else if (cparams.compcode == BLOSC_LZ4HC && nfilter == BLOSC_NOFILTER) {
                            category = 8;
                        }
                        else if (cparams.compcode == BLOSC_LZ4HC && nfilter == BLOSC_SHUFFLE) {
                            category = 9;
                        }
                        else if (cparams.compcode == BLOSC_LZ4HC && nfilter == BLOSC_BITSHUFFLE) {
                            category = 10;
                        }
                        else if (cparams.compcode == BLOSC_LZ4HC && nfilter == BLOSC_FILTER_BYTEDELTA) {
                            category = 11;
                        }
                        else if (cparams.compcode == BLOSC_ZLIB && nfilter == BLOSC_NOFILTER) {
                            category = 12;
                        }
                        else if (cparams.compcode == BLOSC_ZLIB && nfilter == BLOSC_SHUFFLE) {
                            category = 13;
                        }
                        else if (cparams.compcode == BLOSC_ZLIB && nfilter == BLOSC_BITSHUFFLE) {
                            category = 14;
                        }
                        else if (cparams.compcode == BLOSC_ZLIB && nfilter == BLOSC_FILTER_BYTEDELTA) {
                            category = 15;
                        }
                        else if (cparams.compcode == BLOSC_ZSTD && nfilter == BLOSC_NOFILTER) {
                            category = 16;
                        }
                        else if (cparams.compcode == BLOSC_ZSTD && nfilter == BLOSC_SHUFFLE) {
                            category = 17;
                        }
                        else if (cparams.compcode == BLOSC_ZSTD && nfilter == BLOSC_BITSHUFFLE) {
                            category = 18;
                        }
                        else if (cparams.compcode == BLOSC_ZSTD && nfilter == BLOSC_FILTER_BYTEDELTA) {
                            category = 19;
                        }
                    }
                    else if (cparams.splitmode == BLOSC_NEVER_SPLIT) {
                        if (cparams.compcode == BLOSC_BLOSCLZ && nfilter == BLOSC_NOFILTER) {
                            category = 20;
                        }
                        else if (cparams.compcode == BLOSC_BLOSCLZ && nfilter == BLOSC_SHUFFLE) {
                            category = 21;
                        }
                        else if (cparams.compcode == BLOSC_BLOSCLZ && nfilter == BLOSC_BITSHUFFLE) {
                            category = 22;
                        }
                        else if (cparams.compcode == BLOSC_BLOSCLZ && nfilter == BLOSC_FILTER_BYTEDELTA) {
                            category = 23;
                        }
                        else if (cparams.compcode == BLOSC_LZ4 && nfilter == BLOSC_NOFILTER) {
                            category = 24;
                        }
                        else if (cparams.compcode == BLOSC_LZ4 && nfilter == BLOSC_SHUFFLE) {
                            category = 25;
                        }
                        else if (cparams.compcode == BLOSC_LZ4 && nfilter == BLOSC_BITSHUFFLE) {
                            category = 26;
                        }
                        else if (cparams.compcode == BLOSC_LZ4 && nfilter == BLOSC_FILTER_BYTEDELTA) {
                            category = 27;
                        }
                        else if (cparams.compcode == BLOSC_LZ4HC && nfilter == BLOSC_NOFILTER) {
                            category = 28;
                        }
                        else if (cparams.compcode == BLOSC_LZ4HC && nfilter == BLOSC_SHUFFLE) {
                            category = 29;
                        }
                        else if (cparams.compcode == BLOSC_LZ4HC && nfilter == BLOSC_BITSHUFFLE) {
                            category = 30;
                        }
                        else if (cparams.compcode == BLOSC_LZ4HC && nfilter == BLOSC_FILTER_BYTEDELTA) {
                            category = 31;
                        }
                        else if (cparams.compcode == BLOSC_ZLIB && nfilter == BLOSC_NOFILTER) {
                            category = 32;
                        }
                        else if (cparams.compcode == BLOSC_ZLIB && nfilter == BLOSC_SHUFFLE) {
                            category = 33;
                        }
                        else if (cparams.compcode == BLOSC_ZLIB && nfilter == BLOSC_BITSHUFFLE) {
                            category = 34;
                        }
                        else if (cparams.compcode == BLOSC_ZLIB && nfilter == BLOSC_FILTER_BYTEDELTA) {
                            category = 35;
                        }
                        else if (cparams.compcode == BLOSC_ZSTD && nfilter == BLOSC_NOFILTER) {
                            category = 36;
                        }
                        else if (cparams.compcode == BLOSC_ZSTD && nfilter == BLOSC_SHUFFLE) {
                            category = 37;
                        }
                        else if (cparams.compcode == BLOSC_ZSTD && nfilter == BLOSC_BITSHUFFLE) {
                            category = 38;
                        }
                        else if (cparams.compcode == BLOSC_ZSTD && nfilter == BLOSC_FILTER_BYTEDELTA) {
                            category = 39;
                        }
                    }
                    else {
                        category = -1; // Error
                    }

                    fprintf(csv_file, "cratio, speed, special_vals, nchunk, category\n");
                    for (int nchunk = 0; nchunk < schunk->nchunks; nchunk++) {
                        extr_data(csv_file, schunk, nchunk, &cparams, &dparams, category);
                    }

                    fclose(csv_file);
                }
            }
        }
    }
    else {
        // Loop over different filters
        for (int nfilter = 0; nfilter <= FILTER_STOP; ++nfilter) {
            if (nfilter == FILTER_STOP) {
                nfilter = BLOSC_FILTER_BYTEDELTA;
                cparams.filters_meta[BLOSC2_MAX_FILTERS - 1] = schunk->typesize;
                cparams.filters[BLOSC2_MAX_FILTERS - 2] = BLOSC_SHUFFLE;
            }
            cparams.filters[BLOSC2_MAX_FILTERS - 1] = nfilter;
            blosc2_context *cctx = blosc2_create_cctx(cparams);
            blosc2_context *dctx = blosc2_create_dctx(dparams);

            const char *compname = get_compname(cparams.compcode);
            const char *sfilter = get_filtername(nfilter);

            // Create csv file
            sprintf(csv_filename, "%s-%s.csv", compname, sfilter);
            printf("CSV filename: %s\n", csv_filename);
            FILE *csv_file = fopen(csv_filename, "w");
            if (csv_file == NULL) {
                printf("Error creating the file\n");
                return -1;
            }

            int category = -1;
            fprintf(csv_file, "cratio, speed, special_vals, nchunk, category\n");
            for (int nchunk = 0; nchunk < schunk->nchunks; nchunk++) {
                extr_data(csv_file, schunk, nchunk, &cparams, &dparams, category);
            }

            fclose(csv_file);
        }
    }

    /* Free resources */
    blosc2_schunk_free(schunk);
    printf("Success!\n");

    return 0;
}
