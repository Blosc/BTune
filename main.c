#include "btune/btune.h"

#define KB  1024.
#define MB  (1024*KB)

#define CHUNKSIZE (64 * 1024)
#define BLOCKSIZE ( 8 * 1024)

static int fsize(FILE *file) {
    fseek(file, 0, SEEK_END);
    int size = ftell(file);
    fseek(file, 0, SEEK_SET);
    return size;
}

static int get_nchunks_in_file(FILE *file, int chunksize)
{
    int filesize = fsize(file);
    int nchunks = filesize / chunksize;
    if (filesize % CHUNKSIZE != 0) {
        nchunks++;
    }

    return nchunks;
}

int main(int argc, char* argv[])
{
    blosc2_init();

    // Input parameters
    if (argc != 3) {
        fprintf(stderr, "main <input file> <output.b2frame>\n");
        return 1;
    }
    const char* in_fname = argv[1];
    const char* out_fname = argv[2];

    // Open input file
    FILE* finput = fopen(in_fname, "rb");
    if (finput == NULL) {
        fprintf(stderr, "Input file cannot be open.\n");
        return 1;
    }

    // compression params
    blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
    cparams.blocksize = BLOCKSIZE; // If unset there's a division by zero crash

    // btune
    blosc2_btune *btune = malloc(sizeof(blosc2_btune));
    btune_config btune_config = BTUNE_CONFIG_DEFAULTS;
    //btune_config.comp_mode = BTUNE_COMP_HCR;
    //btune_config.behaviour.repeat_mode = BTUNE_REPEAT_ALL;
    btune->btune_config = &btune_config;
    btune->btune_init = btune_init;
    btune->btune_next_blocksize = btune_next_blocksize;
    btune->btune_next_cparams = btune_next_cparams;
    btune->btune_update = btune_update;
    btune->btune_free = btune_free;
    cparams.udbtune = btune;

    // Create super chunk
    remove(out_fname);
    blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;
    blosc2_storage storage = {
        .cparams=&cparams,
        .dparams=&dparams,
        .contiguous=true,
        .urlpath=(char*)out_fname
    };
    blosc2_schunk* schunk = blosc2_schunk_new(&storage);

    // Statistics
    blosc_timestamp_t t0;
    blosc_set_timestamp(&t0);

    // Compress
    int8_t data[CHUNKSIZE];
    int nchunks = get_nchunks_in_file(finput, CHUNKSIZE);
    for (int nchunk = 0; nchunk < nchunks; nchunk++) {
        size_t size = fread(data, 1, CHUNKSIZE, finput);
        if (blosc2_schunk_append_buffer(schunk, data, size) < 0) {
            fprintf(stderr, "Error in appending data to destination file");
            return 1;
        }
    }

    // Statistics
    blosc_timestamp_t t1;
    blosc_set_timestamp(&t1);
    int64_t nbytes = schunk->nbytes;
    int64_t cbytes = schunk->cbytes;
    double ttotal = blosc_elapsed_secs(t0, t1);
    printf("Compression ratio: %.1f MB -> %.1f MB (%.1fx)\n",
            (float)nbytes / MB, (float)cbytes / MB, (1. * (float)nbytes) / (float)cbytes);
    printf("Compression time: %.3g s, %.1f MB/s\n",
            ttotal, (float)nbytes / (ttotal * MB));

    // Free resources
    fclose(finput);
    blosc2_schunk_free(schunk);
    blosc2_destroy();

    return 0;
}
