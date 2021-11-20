# Makefile.

CFLAGS = -O3 -g -Wall
CXX = g++
NVCC = nvcc
INCL=-I/usr/local/cuda/include
NVCCFLAGS = -O3 -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored

FILES = main tinyexr EXRFiles
CUDA_FILES = kernel ReflectanceMapper
CUDA_LIBS = -lcudart -lz

all: $(FILES:=.o) $(CUDA_FILES:=.o)
	$(NVCC) $(NVCCFLAGS) -o reflect $(FILES:=.o) $(CUDA_FILES:=.o) $(CUDA_LIBS)

tinyexr.o: tinyexr/tinyexr.h

.SUFFIXES: .c .cc .cu .o .d
.cc.o:
	$(CXX) $(CFLAGS) $(INCL) -o $@ -c $<

.cu.o:
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

clean:
	rm -f *.o *.d reflect
