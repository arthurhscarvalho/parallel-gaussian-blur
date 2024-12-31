#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char* image_filepath;
    char* output_filepath;
    int num_threads;
    int num_iterations;
    int kernel_size;
    float sigma; // Standard deviation for Gaussian distribution
} Parameters;

const Parameters parse_args(int argc, char* argv[])
{
    Parameters params = { NULL, NULL, 0, 0, 0, 0.0 };
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--image_filepath=", 17) == 0)
            params.image_filepath = argv[i] + 17;
        else if (strncmp(argv[i], "--output_filepath=", 18) == 0)
            params.output_filepath = argv[i] + 18;
        else if (strncmp(argv[i], "--num_threads=", 14) == 0)
            params.num_threads = atoi(argv[i] + 14);
        else if (strncmp(argv[i], "--num_iterations=", 17) == 0)
            params.num_iterations = atoi(argv[i] + 17);
        else if (strncmp(argv[i], "--kernel_size=", 14) == 0)
            params.kernel_size = atof(argv[i] + 14);
        else if (strncmp(argv[i], "--sigma=", 8) == 0)
            params.sigma = atof(argv[i] + 8);
    }
    return params;
}

int validate_parameters(const Parameters* params)
{
    if (!params->image_filepath || !params->output_filepath) {
        printf("Invalid paths passed to argv.\n");
        return 0;
    }
    if (params->kernel_size <= 0 || params->sigma <= 0) {
        printf("Invalid blur parameters passed to argv.\n");
        return 0;
    }
    if (params->num_iterations <= 0) {
        printf("Invalid number of iterations passed to argv.\n");
        return 0;
    }
    if (params->num_threads <= 0) {
        printf("Invalid number of threads passed to argv.\n");
        return 0;
    }
    return 1;
}
