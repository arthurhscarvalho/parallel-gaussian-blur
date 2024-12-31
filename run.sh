#!/bin/bash

IMG_PATH="./assets/image.png"
OUTPUT_PATH="./output/blurred.png"
NUM_THREADS="10"
NUM_ITERATIONS="30"
KERNEL_SIZE="13"
SIGMA="1.0"

mkdir -p output

./process_image.out \
    --image_filepath=$IMG_PATH \
    --output_filepath=$OUTPUT_PATH \
    --num_threads=$NUM_THREADS \
    --num_iterations=$NUM_ITERATIONS \
    --kernel_size=$KERNEL_SIZE \
    --sigma=$SIGMA
