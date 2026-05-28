HIP_PATH?= /opt/rocm/hip
HIPCC=$(HIP_PATH)/bin/hipcc

# these are needed for m5ops
# Assumes gem5 and gem5-resources share a common parent directory.
GEM5_ROOT ?= ../../../..
CFLAGS += -I$(GEM5_ROOT)/include
LDFLAGS += -L$(GEM5_ROOT)/util/m5/build/x86/out -lm5

SRC_DIR := src
BIN_DIR := bin

SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
EXECUTABLES := $(basename $(notdir $(SOURCES)))

.PHONY: all clean

all: $(EXECUTABLES)

$(EXECUTABLES): %: $(SRC_DIR)/%.cpp | $(BIN_DIR)
	$(HIPCC) --amdgpu-target=gfx900,gfx902 $< -o $(BIN_DIR)/$@ $(CFLAGS) $(LDFLAGS)

$(BIN_DIR):
	mkdir -p $@

clean:
	rm -rf $(BIN_DIR)
