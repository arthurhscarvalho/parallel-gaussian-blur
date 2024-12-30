# Define variables
CC = gcc
CFLAGS = -DSTB_IMAGE_IMPLEMENTATION -lm -DSTB_IMAGE_WRITE_IMPLEMENTATION
TARGET = process_image.out
SOURCE = anisotropic_diffusion.c

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SOURCE)
	$(CC) -o $(TARGET) $(SOURCE) $(CFLAGS)

# Clean up
clean:
	rm -f $(TARGET)
