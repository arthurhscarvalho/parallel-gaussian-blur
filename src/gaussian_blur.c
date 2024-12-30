#include "image_io.c"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    const unsigned char* input;
    unsigned char* output;
    int width;
    int height;
    int start_row;
    int end_row;
    int kernel_size;
    const float** kernel;
} ThreadData;

const float** initialize_kernel(int kernel_size)
{
    kernel_size++;
    float** kernel = (float**)malloc(kernel_size * sizeof(float*));
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] = (float*)malloc(kernel_size * sizeof(float));
    }
    kernel_size--;
    float sigma = 1.0;
    float sum = 0.0;
    int half_size = kernel_size / 2;
    // Generate the kernel values
    for (int y = -half_size; y <= half_size; ++y) {
        for (int x = -half_size; x <= half_size; ++x) {
            kernel[y + half_size][x + half_size] = exp(-(x * x + y * y) / (2 * sigma * sigma));
            sum += kernel[y + half_size][x + half_size];
        }
    }
    // Normalize the kernel so that the sum of all elements equals 1
    for (int y = 0; y < kernel_size; ++y) {
        for (int x = 0; x < kernel_size; ++x) {
            kernel[y][x] /= sum;
        }
    }
    return (const float**)kernel;
}

unsigned char clip_to_rgb(float x)
{
    unsigned char clipped = (unsigned char)fminf(255.0f, fmaxf(0.0f, roundf(x)));
    return clipped;
}

void* blur_thread(void* arg)
{
    ThreadData* data = (ThreadData*)arg;
    const float** kernel = data->kernel;
    int offset = data->kernel_size / 2;
    for (int y = data->start_row; y < data->end_row; ++y) {
        for (int x = 0; x < data->width; ++x) {
            for (int c = 0; c < 3; ++c) {
                float sum = 0.0f;
                float weight_sum = 0.0f;
                for (int ky = -offset; ky <= offset; ++ky) {
                    int sy = y + ky;
                    if (sy < 0)
                        sy = 0;
                    if (sy >= data->height)
                        sy = data->height - 1;
                    for (int kx = -offset; kx <= offset; ++kx) {
                        int sx = x + kx;
                        if (sx < 0)
                            sx = 0;
                        if (sx >= data->width)
                            sx = data->width - 1;
                        float k = kernel[ky + offset][kx + offset];
                        sum += data->input[(sy * data->width + sx) * 3 + c] * k;
                        weight_sum += k;
                    }
                }
                pthread_mutex_lock(&mutex);
                data->output[(y * data->width + x) * 3 + c] = clip_to_rgb(sum / weight_sum);
                pthread_mutex_unlock(&mutex);
            }
        }
    }
    return NULL;
}

Image gaussian_blur(const Image image, int kernel_size, const float** kernel, const int num_threads)
{
    int width = image.width;
    int height = image.height;
    Image blurred_image = { NULL, 0, 0 };
    unsigned char* blurred = (unsigned char*)malloc(3 * width * height);
    if (!blurred) {
        return blurred_image;
    }
    memset(blurred, 0, 3 * width * height);
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    // Calculate rows per thread
    int rows_per_thread = height / num_threads;
    // Create threads
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].input = image.data;
        thread_data[i].output = blurred;
        thread_data[i].width = width;
        thread_data[i].height = height;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == num_threads - 1) ? height : (i + 1) * rows_per_thread;
        thread_data[i].kernel = kernel;
        thread_data[i].kernel_size = kernel_size;
        pthread_create(&threads[i], NULL, blur_thread, &thread_data[i]);
    }
    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    blurred_image.data = blurred;
    blurred_image.width = width;
    blurred_image.height = height;
    return blurred_image;
}

void flip_image(unsigned char* image, int width, int height)
{
    int row_size = width * 3;
    unsigned char temp[row_size];
    for (int i = 0; i < height / 2; i++) {
        unsigned char* top_row = image + i * row_size;
        unsigned char* bottom_row = image + (height - 1 - i) * row_size;
        memcpy(temp, top_row, row_size);
        memcpy(top_row, bottom_row, row_size);
        memcpy(bottom_row, temp, row_size);
    }
}

Image apply_gaussian_blur(const Image image, int kernel_size, int num_iterations, int num_threads)
{
    printf("Beginning gaussian blur computation\n");
    struct timeval start, end;
    double elapsed_time;
    gettimeofday(&start, NULL); // Start the timer
    const float** kernel = initialize_kernel(kernel_size);
    Image blurred_image = image;
    for (int i = 0; i < num_iterations; i++) {
        blurred_image = gaussian_blur(blurred_image, kernel_size, kernel, num_threads);
    }
    if (kernel_size % 2 == 0) {
        flip_image(blurred_image.data, blurred_image.width, blurred_image.height);
    }
    gettimeofday(&end, NULL); // Stop the timer
    elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("Finished. Time taken: %.2f seconds\n", elapsed_time);
    return blurred_image;
}
