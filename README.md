# Parallel Gaussian Blur

This project demonstrates the implementation of a **Gaussian Blur** algorithm 
for images using multi-threading in C. While primarily educational, the project 
explores the challenges and solutions in parallelizing image processing tasks.

The Gaussian blur is applied using a full convolution of an RGB image with a 2D 
kernel, ensuring high-quality smoothing across the image. Unlike implementations
optimized for separable kernels (which split the operation into horizontal and 
vertical passes), this project supports non-separable kernels. This makes it a 
practical choice for cases where kernel separability cannot be assumed or 
exploited, providing greater flexibility for specific applications.

Through multi-threading with barrier synchronization and custom kernel creation,
this implementation offers a balance of simplicity and performance.

## Features

- Multi-threaded image processing for better performance;
- Customizable Gaussian kernel parameters (size and sigma);
- Support for multiple iterations of the blur process;
- Image input/output using the lightweight **stb_image** library.

## Directory Structure

```
src/
├── main.c             # Entry point for the program
├── image_io.c         # Handles image reading and writing
├── gaussian_blur.c    # Core Gaussian blur logic and threading
└── argparser.c        # Command-line argument parsing
```

## Prerequisites

- **GCC** (or any C compiler supporting C11 or later)
- **pthread** library for threading
- **stb_image.h** and **stb_image_write.h** libraries for image handling

## Usage

At this moment, this project is aimed for running on Linux.

1. **Clone and compile the project:**

   ```bash
   make clean
   make
   ```
   Alternatively:
   ```bash
   gcc -o process_image.out src/main.c -lm -pthread -DSTB_IMAGE_IMPLEMENTATION -DSTB_IMAGE_WRITE_IMPLEMENTATION
   ```

3. **Run the program:**
   ```bash
   ./process_image.out --image_filepath=<input_image> \
                       --output_filepath=<output_image> \
                       --num_threads=<threads> \
                       --num_iterations=<iterations> \
                       --kernel_size=<size> \
                       --sigma=<sigma>
   ```

    Alternatively, you can edit the file `run.sh` and run it with `bash run.sh`.

   #### Example:
   ```bash
   ./process_image.out --image_filepath=assets/image.png \
                       --output_filepath=./blurred.png \
                       --num_threads=10 \
                       --num_iterations=40 \
                       --kernel_size=15 \
                       --sigma=1.0
   ```

## Parameters

| Parameter          | Description                                   | Example Value       |
|---------------------|-----------------------------------------------|---------------------|
| `--image_filepath`  | Path to the input image                      | `example.png`       |
| `--output_filepath` | Path to save the processed image             | `output.png`        |
| `--num_threads`     | Number of threads to use                     | `8`                 |
| `--num_iterations`  | Number of Gaussian blur iterations to apply  | `20`                 |
| `--kernel_size`     | Size of the Gaussian kernel (must be odd)    | `7`                 |
| `--sigma`           | Standard deviation for the Gaussian kernel   | `1.5`               |

## How It Works

1. **Image Input and Output:** 
   - Reads an image file using `stb_image`.
   - Outputs the processed image as a PNG using `stb_image_write`.

2. **Gaussian Blur Logic:**
   - Creates a normalized Gaussian kernel.
   - Divides the image into chunks for processing by threads.
   - Synchronizes threads using barrier synchronization to ensure correct results.

3. **Parallelization:**
   - Utilizes the `pthread` library for parallel execution.
   - Distributes rows of the image among threads for concurrent processing.

## Example Workflow

<details>
    <summary>Spoiler</summary>
<br>

1. Input Image:  
   ![Input Image](assets/image.png)

2. Output Image (after applying Gaussian blur):  
   ![Output Image](assets/blurred.png)

</details>

## Acknowledgments

- **stb_image** and **stb_image_write** for lightweight image handling.
- `pthread` library for multi-threaded execution.
- Gaussian blur implementation inspired by standard image processing techniques.
- Thanks [Sam Yang](https://x.com/samdoesarts) for the artwork used in the examples!

## License

This project is licensed under the [MIT License](LICENSE).

---

> GitHub [@akaTsunemori](https://github.com/akaTsunemori)
