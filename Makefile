CUDA_PATH ?= /usr/local/cuda

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

NVCCFLAGS     := -g -G -Xcompiler -Wall 

prom = test
src := $(wildcard *.cpp) $(wildcard *.cu)
deps := $(wildcard *.h)
obj := $(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(src)))

%.o: %.cpp $(deps)
	$(NVCC) -c $< -o $@  $(NVCCFLAGS)

%.o: %.cu $(deps)
	$(NVCC) -c $< -o $@  $(NVCCFLAGS) 

$(prom): $(obj)
	$(NVCC) -o $(prom) $(obj)   $(NVCCFLAGS)


clean:
	rm -rf $(obj) $(prom)