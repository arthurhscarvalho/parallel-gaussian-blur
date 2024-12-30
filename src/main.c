#include "gaussian_blur.c"

int main(int argc, char* argv[])
{
    char* image_path = NULL;
    int num_threads = 0;
    int num_iterations = 0;
    int kernel_size = 0;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--image_path=", 13) == 0) {
            image_path = argv[i] + 13;
        } else if (strncmp(argv[i], "--num_threads=", 14) == 0) {
            num_threads = atoi(argv[i] + 14);
        } else if (strncmp(argv[i], "--num_iterations=", 17) == 0) {
            num_iterations = atoi(argv[i] + 17);
        } else if (strncmp(argv[i], "--kernel_size=", 14) == 0) {
            kernel_size = atof(argv[i] + 14);
        }
    }
    if (!image_path || !num_threads || !num_iterations || !kernel_size) {
        printf("Invalid parameters passed to argv.\n");
        return 1;
    }
    Image image = read_image(image_path);
    Image diffused_image = apply_gaussian_blur(image, kernel_size, num_iterations, num_threads);
    write_image(diffused_image.data, diffused_image.width, diffused_image.height, "blurred.png");
    return 0;
}
