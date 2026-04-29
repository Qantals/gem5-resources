
HIP_PATH?= /opt/rocm/hip
HIPCC=$(HIP_PATH)/bin/hipcc

BIN_DIR?= ./bin

.PHONY: square-longrun

square-longrun: $(BIN_DIR)/square-longrun

$(BIN_DIR)/square-longrun: square-longrun.cpp $(BIN_DIR)
	$(HIPCC) --amdgpu-target=gfx900,gfx902 $(CXXFLAGS) square-longrun.cpp -o $(BIN_DIR)/square-longrun

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

.PHONY: square clean
