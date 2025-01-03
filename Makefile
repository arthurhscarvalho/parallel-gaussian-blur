# Define variables
CC = gcc
CFLAGS = -lm -pthread -DSTB_IMAGE_IMPLEMENTATION -DSTB_IMAGE_WRITE_IMPLEMENTATION
TARGET = process_image.out
SOURCE = src/main.c

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SOURCE)
	$(CC) -o $(TARGET) $(SOURCE) $(CFLAGS)

# Clean up
clean:
	rm -f $(TARGET)
