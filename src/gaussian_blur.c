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
    const float* kernel;
} ThreadData;

void initialize_kernel(float* kernel, int kernel_size, float sigma)
{
    float sum = 0.0f;
    int half_size = kernel_size / 2;
    // Generate the kernel values
    for (int y = -half_size; y <= half_size; ++y) {
        for (int x = -half_size; x <= half_size; ++x) {
            kernel[(y + half_size) * kernel_size + (x + half_size)] = exp(-(x * x + y * y) / (2 * sigma * sigma));
            sum += kernel[(y + half_size) * kernel_size + (x + half_size)];
        }
    }
    // Normalize the kernel so that the sum of all elements equals 1
    for (int y = 0; y < kernel_size; ++y) {
        for (int x = 0; x < kernel_size; ++x) {
            kernel[y * kernel_size + x] /= sum;
        }
    }
}

unsigned char clip_to_rgb(float x)
{
    return (unsigned char)fminf(255.0f, fmaxf(0.0f, roundf(x)));
}

void* blur_thread(void* arg)
{
    ThreadData* data = (ThreadData*)arg;
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
                        float k = data->kernel[(ky + offset) * data->kernel_size + (kx + offset)];
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

Image gaussian_blur(const unsigned char* image, int width, int height, int kernel_size, const float* kernel, int num_threads)
{
    Image blurred_image = { NULL, 0, 0 };
    unsigned char* blurred = (unsigned char*)malloc(3 * width * height);
    if (!blurred) {
        return blurred_image;
    }
    memset(blurred, 0, 3 * width * height);
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    int rows_per_thread = height / num_threads;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].input = image;
        thread_data[i].output = blurred;
        thread_data[i].width = width;
        thread_data[i].height = height;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == num_threads - 1) ? height : (i + 1) * rows_per_thread;
        thread_data[i].kernel_size = kernel_size;
        thread_data[i].kernel = kernel;
        pthread_create(&threads[i], NULL, blur_thread, &thread_data[i]);
    }
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

Image apply_gaussian_blur(const unsigned char* image, int width, int height, int kernel_size, int num_iterations, int num_threads)
{
    float kernel[kernel_size * kernel_size];
    initialize_kernel(kernel, kernel_size, 1.0f);
    Image blurred_image = gaussian_blur(image, width, height, kernel_size, kernel, num_threads);
    for (int i = 1; i < num_iterations; i++) {
        blurred_image = gaussian_blur(blurred_image.data, blurred_image.width, blurred_image.height, kernel_size, kernel, num_threads);
    }
    if (kernel_size % 2 == 0) {
        flip_image(blurred_image.data, blurred_image.width, blurred_image.height);
    }
    return blurred_image;
}
