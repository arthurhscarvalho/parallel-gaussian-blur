#include "image_io.c"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

// Barrier implementation remains the same
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
    int total;
} Barrier;

typedef struct {
    const unsigned char* input;
    unsigned char* output;
    unsigned char* temp_buffer;
    int width;
    int height;
    int start_row;
    int end_row;
    int kernel_size;
    const float** kernel;
    Barrier* barrier;
    int total_iterations;
} ThreadData;

void barrier_init(Barrier* barrier, int total)
{
    pthread_mutex_init(&barrier->mutex, NULL);
    pthread_cond_init(&barrier->cond, NULL);
    barrier->count = 0;
    barrier->total = total;
}

void barrier_wait(Barrier* barrier)
{
    pthread_mutex_lock(&barrier->mutex);
    barrier->count++;
    if (barrier->count == barrier->total) {
        barrier->count = 0;
        pthread_cond_broadcast(&barrier->cond);
    } else {
        while (barrier->count > 0) {
            pthread_cond_wait(&barrier->cond, &barrier->mutex);
        }
    }
    pthread_mutex_unlock(&barrier->mutex);
}

// Kernel initialization remains the same
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
    for (int y = -half_size; y <= half_size; ++y) {
        for (int x = -half_size; x <= half_size; ++x) {
            kernel[y + half_size][x + half_size] = exp(-(x * x + y * y) / (2 * sigma * sigma));
            sum += kernel[y + half_size][x + half_size];
        }
    }
    for (int y = 0; y < kernel_size; ++y) {
        for (int x = 0; x < kernel_size; ++x) {
            kernel[y][x] /= sum;
        }
    }
    return (const float**)kernel;
}

unsigned char clip_to_rgb(float x)
{
    return (unsigned char)fminf(255.0f, fmaxf(0.0f, roundf(x)));
}

void process_chunk(const unsigned char* input, unsigned char* output, ThreadData* data)
{
    int offset = data->kernel_size / 2;
    for (int y = data->start_row; y < data->end_row; ++y) {
        for (int x = 0; x < data->width; ++x) {
            float sums[3] = { 0.0f, 0.0f, 0.0f };
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
                    float k = data->kernel[ky + offset][kx + offset];
                    const unsigned char* pixel = &input[(sy * data->width + sx) * 3];
                    sums[0] += pixel[0] * k;
                    sums[1] += pixel[1] * k;
                    sums[2] += pixel[2] * k;
                    weight_sum += k;
                }
            }
            unsigned char* out_pixel = &output[(y * data->width + x) * 3];
            out_pixel[0] = clip_to_rgb(sums[0] / weight_sum);
            out_pixel[1] = clip_to_rgb(sums[1] / weight_sum);
            out_pixel[2] = clip_to_rgb(sums[2] / weight_sum);
        }
    }
}

void* blur_thread(void* arg)
{
    ThreadData* data = (ThreadData*)arg;
    // Copy initial input to temp buffer
    if (data->start_row == 0) { // Only one thread needs to do this
        memcpy(data->temp_buffer, data->input, data->width * data->height * 3);
    }
    barrier_wait(data->barrier); // Ensure copy is complete before starting
    for (int iter = 0; iter < data->total_iterations; iter++) {
        // For even iterations, read from temp and write to output
        // For odd iterations, read from output and write to temp
        if (iter % 2 == 0) {
            process_chunk(data->temp_buffer, data->output, data);
        } else {
            process_chunk(data->output, data->temp_buffer, data);
        }
        barrier_wait(data->barrier); // Wait for all threads before next iteration
    }
    // If we ended on an odd iteration, copy temp buffer to output
    if (data->total_iterations % 2 != 0 && data->start_row == 0) {
        memcpy(data->output, data->temp_buffer, data->width * data->height * 3);
    }
    return NULL;
}

Image compute_gaussian_blur(const Image image, int kernel_size, int num_threads, int num_iterations)
{
    const float** kernel = initialize_kernel(kernel_size);
    Image blurred = { NULL, image.width, image.height };
    unsigned char* output = malloc(3 * image.width * image.height);
    unsigned char* temp = malloc(3 * image.width * image.height);
    if (!output || !temp) {
        free(output);
        free(temp);
        return blurred;
    }
    Barrier barrier;
    barrier_init(&barrier, num_threads);
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    ThreadData* thread_data = malloc(num_threads * sizeof(ThreadData));
    int rows_per_thread = image.height / num_threads;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].input = image.data;
        thread_data[i].output = output;
        thread_data[i].temp_buffer = temp;
        thread_data[i].width = image.width;
        thread_data[i].height = image.height;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == num_threads - 1) ? image.height : (i + 1) * rows_per_thread;
        thread_data[i].kernel = kernel;
        thread_data[i].kernel_size = kernel_size;
        thread_data[i].barrier = &barrier;
        thread_data[i].total_iterations = num_iterations;
        pthread_create(&threads[i], NULL, blur_thread, &thread_data[i]);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    // Clean up
    free(threads);
    free(thread_data);
    free(temp);
    // Set up result
    blurred.data = output;
    return blurred;
}

Image apply_gaussian_blur(const Image image, int kernel_size, int num_iterations, int num_threads)
{
    printf("Beginning gaussian blur computation\n");
    struct timeval start, end;
    gettimeofday(&start, NULL);
    Image blurred = compute_gaussian_blur(image, kernel_size, num_threads, num_iterations);
    gettimeofday(&end, NULL);
    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("Finished. Time taken: %.2f seconds\n", elapsed_time);
    return blurred;
}