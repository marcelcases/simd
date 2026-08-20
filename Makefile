# Makefile for scalar and std::simd implementations.
# Usage: make; make scalar; make simd; make run; make riscv

BUILD_DIR := build
EXERCISES := 01_add 02_sum 03_clamp 04_count 05_softmax 06_fma 07_filter 08_conv1d
SCALAR_TARGETS := $(addprefix $(BUILD_DIR)/,$(addsuffix _scalar,$(EXERCISES)))
SIMD_TARGETS := $(addprefix $(BUILD_DIR)/,$(addsuffix _simd,$(EXERCISES)))
RISCV_TARGETS := $(addprefix $(BUILD_DIR)/,$(addsuffix _simd.riscv,$(EXERCISES)))

# Detect platform
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
    # macOS: requires Homebrew GCC for std::experimental::simd.
    CXX = g++-15
    ifeq ($(UNAME_M),arm64)
        CXXFLAGS = -std=c++2b -O3 -mcpu=apple-m1 -fno-math-errno -fno-trapping-math -Wall -Wextra -Isrc
    else
        CXXFLAGS = -std=c++2b -O3 -march=native -fno-math-errno -fno-trapping-math -Wall -Wextra -Isrc
    endif
else
    # Linux / MareNostrum 5.
    CXX = g++
    CXXFLAGS = -std=c++2b -O3 -march=native -mavx512f -mavx512vl -mavx512dq -mavx512bw -fno-math-errno -fno-trapping-math -Wall -Wextra -Isrc
endif

# RISC-V cross-compilation settings for the SIMD implementations.
RISCV_CXX = riscv64-linux-gnu-g++
RISCV_CXXFLAGS = -std=c++2b -O3 -march=rv64gcv -static -fno-math-errno -fno-trapping-math -Wall -Wextra -Isrc
RISCV_LDLIBS = -lm
QEMU_RISCV = qemu-riscv64-static

LDLIBS = -lm

all: scalar simd

scalar: $(SCALAR_TARGETS)

simd: $(SIMD_TARGETS)

$(BUILD_DIR)/%_scalar: src/scalar/%.cpp src/common.h Makefile
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(LDLIBS) -o $@

$(BUILD_DIR)/%_simd: src/simd/%.cpp src/common.h src/simd_common.h Makefile
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(LDLIBS) -o $@

run: run-scalar run-simd

run-scalar: scalar
	@for ex in $(EXERCISES); do echo "\n=== scalar/$$ex ==="; ./$(BUILD_DIR)/$${ex}_scalar; done

run-simd: simd
	@for ex in $(EXERCISES); do echo "\n=== simd/$$ex ==="; ./$(BUILD_DIR)/$${ex}_simd; done

# RISC-V targets use only the explicit SIMD implementations.
riscv: $(RISCV_TARGETS)

$(BUILD_DIR)/%_simd.riscv: src/simd/%.cpp src/common.h src/simd_common.h Makefile
	@mkdir -p $(BUILD_DIR)
	$(RISCV_CXX) $(RISCV_CXXFLAGS) $< $(RISCV_LDLIBS) -o $@

run-riscv-128: $(RISCV_TARGETS)
	@echo "=== Running SIMD examples with RISC-V VLEN=128 ==="
	@for ex in $(EXERCISES); do \
		echo "\n=== $$ex (VLEN=128) ==="; \
		$(QEMU_RISCV) -cpu rv64,v=true,vlen=128 ./$(BUILD_DIR)/$${ex}_simd.riscv; \
	done

run-riscv-512: $(RISCV_TARGETS)
	@echo "=== Running SIMD examples with RISC-V VLEN=512 ==="
	@for ex in $(EXERCISES); do \
		echo "\n=== $$ex (VLEN=512) ==="; \
		$(QEMU_RISCV) -cpu rv64,v=true,vlen=512 ./$(BUILD_DIR)/$${ex}_simd.riscv; \
	done

run-riscv-both: $(RISCV_TARGETS)
	@$(MAKE) run-riscv-128
	@echo "\n========================================\n"
	@$(MAKE) run-riscv-512

verify-riscv: $(BUILD_DIR)/01_add_simd.s
	@echo "=== Checking for RISC-V Vector (RVV) instructions ==="
	@grep -E 'vle32\.v|vse32\.v|vfadd\.vv|vfmul\.vv|vfmacc\.vv' $< || echo "No RVV instructions found - check compiler flags"

$(BUILD_DIR)/01_add_simd.s: src/simd/01_add.cpp src/common.h src/simd_common.h Makefile
	@mkdir -p $(BUILD_DIR)
	$(RISCV_CXX) $(RISCV_CXXFLAGS) -S $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all scalar simd run run-scalar run-simd riscv run-riscv-128 run-riscv-512 run-riscv-both verify-riscv clean
