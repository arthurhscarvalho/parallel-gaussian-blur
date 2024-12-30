#include "stb_image.h"
#include "stb_image_write.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define NUM_THREADS 10
#define NUM_ITERATIONS 10
#define K 20.0f // Constant controlling edge preservation

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

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

float compute_diffusion_coefficient(float gradient_magnitude)
{
    return exp(-(gradient_magnitude * gradient_magnitude) / (K * K));
}

unsigned char clip_to_rgb(float x)
{
    unsigned char clipped = (unsigned char)fminf(255.0f, fmaxf(0.0f, roundf(x)));
    return clipped;
}

void* diffusion_thread(void* arg)
{
    ThreadData* data = (ThreadData*)arg;
    for (int y = data->start_row; y < data->end_row; ++y) {
        for (int x = 0; x < data->width; ++x) {
            for (int c = 0; c < 3; ++c) {
                float sum = 0.0f;
                float weight_sum = 0.0f;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int sx = x + kx;
                        int sy = y + ky;
                        if (sx < 0 || sx >= data->width || sy < 0 || sy >= data->height)
                            continue;

                        // Calculate the gradient magnitude
                        float diff_x = data->input[(sy * data->width + sx) * 3 + c] - data->input[(y * data->width + x) * 3 + c];
                        float diff_y = data->input[((sy + 1) * data->width + sx) * 3 + c] - data->input[((y + 1) * data->width + x) * 3 + c];
                        float gradient_magnitude = sqrt(diff_x * diff_x + diff_y * diff_y);

                        // Compute the diffusion coefficient
                        float diff_coeff = compute_diffusion_coefficient(gradient_magnitude);

                        // Apply the diffusion
                        sum += data->input[(sy * data->width + sx) * 3 + c] * diff_coeff;
                        weight_sum += diff_coeff;
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

Image anisotropic_diffusion(const unsigned char* image, int width, int height)
{
    Image diffused_image = { NULL, 0, 0 };
    unsigned char* diffused = (unsigned char*)malloc(3 * width * height);
    if (!diffused) {
        return diffused_image;
    }
    memcpy(diffused, image, 3 * width * height);

    const int num_threads = NUM_THREADS;
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    // Calculate rows per thread
    int rows_per_thread = height / num_threads;

    // Perform diffusion for the specified number of iterations
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // Create threads for each iteration
        for (int i = 0; i < num_threads; i++) {
            thread_data[i].input = diffused;
            thread_data[i].output = diffused;
            thread_data[i].width = width;
            thread_data[i].height = height;
            thread_data[i].start_row = i * rows_per_thread;
            thread_data[i].end_row = (i == num_threads - 1) ? height : (i + 1) * rows_per_thread;
            pthread_create(&threads[i], NULL, diffusion_thread, &thread_data[i]);
        }

        // Wait for all threads to complete
        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }
    }

    diffused_image.data = diffused;
    diffused_image.width = width;
    diffused_image.height = height;
    return diffused_image;
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
    Image diffused_image = anisotropic_diffusion(image.data, image.width, image.height);
    write_image(diffused_image.data, diffused_image.width, diffused_image.height, "diffused.png");
    return 0;
}
