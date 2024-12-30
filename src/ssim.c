#include <float.h>
#include <math.h>

float ssim(const unsigned char* reference_image, const unsigned char* test_image, int width, int height)
{
    const float C1 = 0.01f * 0.01f;
    const float C2 = 0.03f * 0.03f;
    float ssim_value = 0.0f;
    int window_size = 11; // 11x11 window
    int window_half = window_size / 2;
    for (int y = window_half; y < height - window_half; y++) {
        for (int x = window_half; x < width - window_half; x++) {
            // Compute local mean for reference and test images
            float mu_x = 0.0f, mu_y = 0.0f;
            float sigma_x = 0.0f, sigma_y = 0.0f, sigma_xy = 0.0f;
            for (int wy = -window_half; wy <= window_half; wy++) {
                for (int wx = -window_half; wx <= window_half; wx++) {
                    int ref_pixel = reference_image[(y + wy) * width + (x + wx)];
                    int test_pixel = test_image[(y + wy) * width + (x + wx)];

                    mu_x += ref_pixel;
                    mu_y += test_pixel;
                }
            }
            mu_x /= (window_size * window_size);
            mu_y /= (window_size * window_size);
            // Compute variance and covariance
            for (int wy = -window_half; wy <= window_half; wy++) {
                for (int wx = -window_half; wx <= window_half; wx++) {
                    int ref_pixel = reference_image[(y + wy) * width + (x + wx)];
                    int test_pixel = test_image[(y + wy) * width + (x + wx)];

                    sigma_x += (ref_pixel - mu_x) * (ref_pixel - mu_x);
                    sigma_y += (test_pixel - mu_y) * (test_pixel - mu_y);
                    sigma_xy += (ref_pixel - mu_x) * (test_pixel - mu_y);
                }
            }
            sigma_x /= (window_size * window_size);
            sigma_y /= (window_size * window_size);
            sigma_xy /= (window_size * window_size);
            // Compute SSIM for the current window
            float numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2);
            float denominator = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2);
            ssim_value += numerator / denominator;
        }
    }
    // Normalize SSIM value by the number of windows
    ssim_value /= (float)((width - window_size) * (height - window_size));
    return ssim_value;
}
