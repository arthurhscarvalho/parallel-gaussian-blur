#include "image_io.c"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

/**
 * Barrier synchronization structure for thread coordination
 */
typedef struct {
    pthread_mutex_t mutex; // Mutex for thread synchronization
    pthread_cond_t cond; // Condition variable for signaling
    int count; // Current count of threads at barrier
    int total; // Total number of threads to wait for
} Barrier;

/**
 * Thread data structure containing all necessary parameters for blur operation
 */
typedef struct {
    const unsigned char* input; // Input image data
    unsigned char* output; // Output image data
    unsigned char* temp_buffer; // Temporary buffer for intermediate results
    int width; // Image width
    int height; // Image height
    int start_row; // Starting row for this thread
    int end_row; // Ending row for this thread
    int kernel_size; // Size of the Gaussian kernel
    const float** kernel; // Gaussian kernel coefficients
    Barrier* barrier; // Barrier for thread synchronization
    int total_iterations; // Number of blur iterations to perform
} ThreadData;

/**
 * Initializes a barrier synchronization object
 * @param barrier Pointer to barrier structure
 * @param total Total number of threads that will use this barrier
 */
void barrier_init(Barrier* barrier, int total)
{
    pthread_mutex_init(&barrier->mutex, NULL);
    pthread_cond_init(&barrier->cond, NULL);
    barrier->count = 0;
    barrier->total = total;
}

/**
 * Implements barrier synchronization
 * All threads must call this function before any can proceed
 * @param barrier Pointer to barrier structure
 */
void barrier_wait(Barrier* barrier)
{
    pthread_mutex_lock(&barrier->mutex);
    barrier->count++;
    if (barrier->count == barrier->total) {
        // Last thread arrives, reset count and wake all threads
        barrier->count = 0;
        pthread_cond_broadcast(&barrier->cond);
    } else {
        // Wait for all threads to arrive
        while (barrier->count > 0) {
            pthread_cond_wait(&barrier->cond, &barrier->mutex);
        }
    }
    pthread_mutex_unlock(&barrier->mutex);
}

/**
 * Creates and initializes a Gaussian kernel
 * @param kernel_size Size of the kernel (must be odd)
 * @return 2D array containing the normalized Gaussian kernel
 */
const float** initialize_kernel(int kernel_size)
{
    float** kernel = (float**)malloc(kernel_size * sizeof(float*));
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] = (float*)malloc(kernel_size * sizeof(float));
    }
    float sigma = 1.0; // Standard deviation for Gaussian distribution
    float sum = 0.0; // For normalization
    int half_size = kernel_size / 2;
    // Calculate Gaussian values
    for (int y = -half_size; y <= half_size; ++y) {
        for (int x = -half_size; x <= half_size; ++x) {
            kernel[y + half_size][x + half_size] = exp(-(x * x + y * y) / (2 * sigma * sigma));
            sum += kernel[y + half_size][x + half_size];
        }
    }
    // Normalize the kernel
    for (int y = 0; y < kernel_size; ++y) {
        for (int x = 0; x < kernel_size; ++x) {
            kernel[y][x] /= sum;
        }
    }
    return (const float**)kernel;
}

/**
 * Clips a float value to valid RGB range (0-255)
 * @param x Input float value
 * @return Clipped unsigned char value
 */
unsigned char clip_to_rgb(float x)
{
    return (unsigned char)fminf(255.0f, fmaxf(0.0f, roundf(x)));
}

/**
 * Processes a chunk of the image applying Gaussian blur
 * @param input Input image data
 * @param output Output image data
 * @param data Thread-specific data containing processing parameters
 */
void process_chunk(const unsigned char* input, unsigned char* output, ThreadData* data)
{
    int offset = data->kernel_size / 2;
    // Process each pixel in the assigned chunk
    for (int y = data->start_row; y < data->end_row; ++y) {
        for (int x = 0; x < data->width; ++x) {
            float sums[3] = { 0.0f, 0.0f, 0.0f };
            float weight_sum = 0.0f;
            // Apply kernel to neighborhood
            for (int ky = -offset; ky <= offset; ++ky) {
                int sy = y + ky;
                // Handle border cases with clamping
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
                    // Accumulate weighted values for each color channel
                    sums[0] += pixel[0] * k;
                    sums[1] += pixel[1] * k;
                    sums[2] += pixel[2] * k;
                    weight_sum += k;
                }
            }
            // Write output pixel
            unsigned char* out_pixel = &output[(y * data->width + x) * 3];
            out_pixel[0] = clip_to_rgb(sums[0] / weight_sum);
            out_pixel[1] = clip_to_rgb(sums[1] / weight_sum);
            out_pixel[2] = clip_to_rgb(sums[2] / weight_sum);
        }
    }
}

/**
 * Thread function for parallel Gaussian blur processing
 * @param arg Pointer to thread-specific data
 * @return NULL
 */
void* blur_thread(void* arg)
{
    ThreadData* data = (ThreadData*)arg;
    // First thread copies input to temp buffer
    if (data->start_row == 0) {
        memcpy(data->temp_buffer, data->input, data->width * data->height * 3);
    }
    barrier_wait(data->barrier);
    // Perform multiple iterations of blur
    for (int iter = 0; iter < data->total_iterations; iter++) {
        // Alternate between buffers for each iteration
        if (iter % 2 == 0) {
            process_chunk(data->temp_buffer, data->output, data);
        } else {
            process_chunk(data->output, data->temp_buffer, data);
        }
        barrier_wait(data->barrier);
    }
    // If odd number of iterations, copy temp buffer to output
    if (data->total_iterations % 2 != 0 && data->start_row == 0) {
        memcpy(data->output, data->temp_buffer, data->width * data->height * 3);
    }
    return NULL;
}

/**
 * Main function to compute Gaussian blur using multiple threads.
 * The kernel size must be odd. If an even value is passed, it will be
 * incremented so that it's an odd number. This is a common behavior
 * in gaussian kernels.
 * @param image Input image structure
 * @param kernel_size Size of the Gaussian kernel
 * @param num_threads Number of threads to use
 * @param num_iterations Number of blur iterations to perform
 * @return Blurred image structure
 */
Image compute_gaussian_blur(const Image image, int kernel_size, int num_threads, int num_iterations)
{
    if (kernel_size % 2 == 0)
        kernel_size++;
    const float** kernel = initialize_kernel(kernel_size);
    Image blurred = { NULL, image.width, image.height };
    // Allocate output and temporary buffers
    unsigned char* output = malloc(3 * image.width * image.height);
    unsigned char* temp = malloc(3 * image.width * image.height);
    if (!output || !temp) {
        free(output);
        free(temp);
        return blurred;
    }
    // Initialize thread synchronization
    Barrier barrier;
    barrier_init(&barrier, num_threads);
    // Create and initialize threads
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    ThreadData* thread_data = malloc(num_threads * sizeof(ThreadData));
    int rows_per_thread = image.height / num_threads;
    // Launch threads
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
    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    // Cleanup and return data
    free(threads);
    free(thread_data);
    free(temp);
    blurred.data = output;
    return blurred;
}

/**
 * Wrapper function that measures and reports execution time
 * @param image Input image structure
 * @param kernel_size Size of the Gaussian kernel
 * @param num_iterations Number of blur iterations
 * @param num_threads Number of threads to use
 * @return Blurred image structure
 */
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