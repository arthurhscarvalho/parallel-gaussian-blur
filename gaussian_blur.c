#include "stb_image.h"
#include "stb_image_write.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define NUM_THREADS 10
#define KERNEL_SIZE 7
#define NUM_ITERATIONS 5

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
float kernel[KERNEL_SIZE][KERNEL_SIZE];

typedef struct {
    unsigned char* data;
    int width;
    int height;
} Image;

typedef struct {
    const unsigned char* input;
    unsigned char* output;
    int width;
    int height;
    int start_row;
    int end_row;
} ThreadData;

void initialize_kernel()
{
    float sigma = 1.0f;
    float sum = 0.0f;
    int half_size = KERNEL_SIZE / 2;
    // Generate the kernel values
    for (int y = -half_size; y <= half_size; ++y) {
        for (int x = -half_size; x <= half_size; ++x) {
            kernel[y + half_size][x + half_size] = exp(-(x * x + y * y) / (2 * sigma * sigma));
            sum += kernel[y + half_size][x + half_size];
        }
    }
    // Normalize the kernel so that the sum of all elements equals 1
    for (int y = 0; y < KERNEL_SIZE; ++y) {
        for (int x = 0; x < KERNEL_SIZE; ++x) {
            kernel[y][x] /= sum;
        }
    }
}

unsigned char clip_to_rgb(float x)
{
    unsigned char clipped = (unsigned char)fminf(255.0f, fmaxf(0.0f, roundf(x)));
    return clipped;
}

void* blur_thread(void* arg)
{
    ThreadData* data = (ThreadData*)arg;
    int offset = KERNEL_SIZE / 2;
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

Image gaussian_blur(const unsigned char* image, int width, int height)
{
    initialize_kernel();
    Image blurred_image = { NULL, 0, 0 };
    unsigned char* blurred = (unsigned char*)malloc(3 * width * height);
    if (!blurred) {
        return blurred_image;
    }
    memset(blurred, 0, 3 * width * height);
    const int num_threads = NUM_THREADS;
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    // Calculate rows per thread
    int rows_per_thread = height / num_threads;
    // Create threads
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].input = image;
        thread_data[i].output = blurred;
        thread_data[i].width = width;
        thread_data[i].height = height;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == num_threads - 1) ? height : (i + 1) * rows_per_thread;
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

Image apply_gaussian_blur(const unsigned char* image, int width, int height)
{
    Image blurred_image = gaussian_blur(image, width, height);
    for (int i = 1; i < NUM_ITERATIONS; i++) {
        blurred_image = gaussian_blur(blurred_image.data, blurred_image.width, blurred_image.height);
    }
    if (KERNEL_SIZE % 2 == 0) {
        flip_image(blurred_image.data, blurred_image.width, blurred_image.height);
    }
    return blurred_image;
}

Image read_image(const char* image_path)
{
    Image image = { NULL, 0, 0 };
    int width, height, channels;
    // Load the image with stb_image (force grayscale by passing 1 as desired_channels)
    unsigned char* loaded_image = stbi_load(image_path, &width, &height, &channels, 3);
    if (!loaded_image) {
        printf("Failed to load image: %s\n", stbi_failure_reason());
        return image;
    }
    printf("Image loaded successfully!\n");
    printf("Dimensions: %dx%d\n", width, height);
    image.data = loaded_image;
    image.width = width;
    image.height = height;
    return image;
}

int write_image(const unsigned char* image, int width, int height, const char* filepath)
{
    if (!image || width <= 0 || height <= 0 || !filepath) {
        printf("Invalid input parameters for writing the image.\n");
        return 0;
    }
    // Save the image as PNG
    if (stbi_write_png(filepath, width, height, 3, image, width * 3) == 0) {
        printf("Failed to save the image to %s\n", filepath);
        return 0;
    }
    printf("Image saved successfully to %s\n", filepath);
    return 1;
}

int main(int argc, char* argv[])
{
    const char* image_path = argv[1];
    Image image = read_image(image_path);
    Image blurred_image = apply_gaussian_blur(image.data, image.width, image.height);
    write_image(blurred_image.data, blurred_image.width, blurred_image.height, "blurred.png");
    return 0;
}
