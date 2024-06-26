# Compiler
CC = nvcc

# Include directories
INCLUDES = -I/usr/local/cuda/include -I/usr/local/TensorRT/include -I/usr/local/include/opencv4/ -I/usr/local/include/jetson-utils/ -I/usr/local/include/jetson-inference/

# Library directories
LIBS = -L/usr/local/cuda/lib64 -L/usr/local/TensorRT/lib -L/usr/local/lib

# Libraries to link against
LINK = -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lnvinfer -lcudart -lcuda -ljetson-utils -ljetson-inference

# Source files
SRCS = tensorrt_segformer_inference.cpp

# Output executable
OUT = tensorrt_segformer_inference

all:
	$(CC) $(SRCS) -o $(OUT) $(INCLUDES) $(LIBS) $(LINK)

clean:
	rm -f $(OUT)
