#!/bin/bash

IMG_PATH="./assets/image.png"
NUM_THREADS="10"
KERNEL_SIZE="7"
NUM_ITERATIONS="10"

./process_image.out \
    --image_path=$IMG_PATH \
    --kernel_size=$KERNEL_SIZE \
    --num_threads=$NUM_THREADS \
    --num_iterations=$NUM_ITERATIONS
