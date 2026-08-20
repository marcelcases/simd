# Makefile for std::simd examples
# Cross-platform: Intel AVX-512 (MareNostrum 5), Apple Silicon (NEON), and RISC-V RVV
# Usage: make; make run; make riscv; make run-riscv-128

BUILD_DIR := build
EXAMPLES := 01_add 02_sum 03_clamp 04_count 05_softmax 06_fma 07_filter 08_conv1d
TARGETS := $(addprefix $(BUILD_DIR)/,$(EXAMPLES))
TARGETS_RISCV := $(addsuffix .riscv,$(TARGETS))

# Detect platform
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
    # macOS - requires Homebrew GCC for std::experimental::simd
    # Install with: brew install gcc
    CXX = g++-15
    ifeq ($(UNAME_M),arm64)
        # Apple Silicon (M1/M2/M3) - ARM NEON (128-bit vectors = 4 floats)
        CXXFLAGS = -std=c++2b -O3 -mcpu=apple-m1 -fno-math-errno -fno-trapping-math -Wall -Wextra -Isrc
    else
        # Intel Mac
        CXXFLAGS = -std=c++2b -O3 -march=native -fno-math-errno -fno-trapping-math -Wall -Wextra -Isrc
    endif
else
    # Linux (MareNostrum 5 / Intel AVX-512)
    CXX = g++
    CXXFLAGS = -std=c++2b -O3 -march=native -mavx512f -mavx512vl -mavx512dq -mavx512bw -fno-math-errno -fno-trapping-math -Wall -Wextra -Isrc
endif

# RISC-V cross-compilation settings
RISCV_CXX = riscv64-linux-gnu-g++
RISCV_CXXFLAGS = -std=c++2b -O3 -march=rv64gcv -static -fno-math-errno -fno-trapping-math -Wall -Wextra -Isrc
RISCV_LDLIBS = -lm
QEMU_RISCV = qemu-riscv64-static

LDLIBS = -lm

all: $(TARGETS)

run: $(TARGETS)
	@for ex in $(EXAMPLES); do echo "\n=== $$ex ==="; ./$(BUILD_DIR)/$$ex; done

$(BUILD_DIR)/%: src/%.cpp src/common.h Makefile
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(LDLIBS) -o $@

# RISC-V targets
riscv: $(TARGETS_RISCV)

$(BUILD_DIR)/%.riscv: src/%.cpp src/common.h Makefile
	@mkdir -p $(BUILD_DIR)
	$(RISCV_CXX) $(RISCV_CXXFLAGS) $< $(RISCV_LDLIBS) -o $@

# Run RISC-V binaries with VLEN=128 (4 floats, like ARM NEON)
run-riscv-128: $(TARGETS_RISCV)
	@echo "=== Running RISC-V examples with VLEN=128 (4 floats per register) ==="
	@for ex in $(EXAMPLES); do \
		echo "\n=== $$ex (VLEN=128) ==="; \
		$(QEMU_RISCV) -cpu rv64,v=true,vlen=128 ./$(BUILD_DIR)/$$ex.riscv; \
	done

# Run RISC-V binaries with VLEN=512 (16 floats, like AVX-512)
run-riscv-512: $(TARGETS_RISCV)
	@echo "=== Running RISC-V examples with VLEN=512 (16 floats per register) ==="
	@for ex in $(EXAMPLES); do \
		echo "\n=== $$ex (VLEN=512) ==="; \
		$(QEMU_RISCV) -cpu rv64,v=true,vlen=512 ./$(BUILD_DIR)/$$ex.riscv; \
	done

# Run both VLEN configurations for comparison
run-riscv-both: $(TARGETS_RISCV)
	@$(MAKE) run-riscv-128
	@echo "\n========================================\n"
	@$(MAKE) run-riscv-512

# Generate assembly for RISC-V to verify vector instructions
verify-riscv: $(BUILD_DIR)/01_add.s
	@echo "=== Checking for RISC-V Vector (RVV) instructions ==="
	@grep -E 'vle32\.v|vse32\.v|vfadd\.vv|vfmul\.vv|vfmacc\.vv' $< || echo "No RVV instructions found - check compiler flags"

$(BUILD_DIR)/01_add.s: src/01_add.cpp src/common.h Makefile
	@mkdir -p $(BUILD_DIR)
	$(RISCV_CXX) $(RISCV_CXXFLAGS) -S $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all run riscv run-riscv-128 run-riscv-512 run-riscv-both verify-riscv clean
