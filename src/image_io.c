#include "../lib/stb_image.h"
#include "../lib/stb_image_write.h"

typedef struct {
    unsigned char* data;
    int width;
    int height;
} Image;

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