CUDA_PATH ?= /usr/local/cuda

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

NVCCFLAGS     := -g -G -Xcompiler -Iinclude -Isrc

prom = test
src := $(wildcard src/*.cpp) $(wildcard src/*.cu)
deps := $(wildcard include/*.h)
obj := $(patsubst src/%.cpp,build/%.o,$(patsubst src/%.cu,build/%.o,$(src)))


# # 将源文件转换为目标文件
# obj := $(patsubst src/%,build/%,$(src:.cpp=.o)) $(patsubst src/%,build/%,$(src:.cu=.o))


# 定义输出目录
OBJDIR = build

# # 创建输出目录
# $(OBJDIR):
# 	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: src/%.cpp $(deps) | $(OBJDIR)
	$(NVCC) -c $< -o $@  $(NVCCFLAGS)

$(OBJDIR)/%.o: src/%.cu $(deps) | $(OBJDIR)
	$(NVCC) -c $< -o $@  $(NVCCFLAGS) 

$(prom): $(obj)
	$(NVCC) -o $(prom) $(obj)   $(NVCCFLAGS)


clean:
	rm -rf $(OBJDIR)/*.o $(prom)