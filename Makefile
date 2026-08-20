# Scalar and std::simd algorithms with separate benchmark drivers.

BUILD_DIR := build
EXERCISES := 01_add 02_sum 03_clamp 04_count 05_softmax 06_fma 07_filter 08_conv1d
BENCH_TARGETS := $(addprefix $(BUILD_DIR)/,$(addsuffix _bench,$(EXERCISES)))
RISCV_TARGETS := $(addprefix $(BUILD_DIR)/,$(addsuffix _bench.riscv,$(EXERCISES)))

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
    # macOS: requires Homebrew GCC for std::experimental::simd.
    CXX = g++-15
    ifeq ($(UNAME_M),arm64)
        CXXFLAGS = -std=c++2b -O3 -mcpu=apple-m1 -fno-math-errno -fno-trapping-math -Wall -Wextra -Ibench -Iinclude -Isrc
    else
        CXXFLAGS = -std=c++2b -O3 -march=native -fno-math-errno -fno-trapping-math -Wall -Wextra -Ibench -Iinclude -Isrc
    endif
else
    # Linux / MareNostrum 5.
    CXX = g++
    CXXFLAGS = -std=c++2b -O3 -march=native -mavx512f -mavx512vl -mavx512dq -mavx512bw -fno-math-errno -fno-trapping-math -Wall -Wextra -Ibench -Iinclude -Isrc
endif

# RISC-V cross-compilation settings.
RISCV_CXX = riscv64-linux-gnu-g++
RISCV_CXXFLAGS = -std=c++2b -O3 -march=rv64gcv -static -fno-math-errno -fno-trapping-math -Wall -Wextra -Ibench -Iinclude -Isrc
RISCV_LDLIBS = -lm
QEMU_RISCV = qemu-riscv64-static

LDLIBS = -lm

all: bench

bench: $(BENCH_TARGETS)

$(BUILD_DIR)/%_bench: bench/%.cpp src/scalar/%.cpp src/simd/%.cpp include/simd_examples/%.hpp bench/benchmark_common.hpp src/simd_common.h Makefile
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) bench/$*.cpp src/scalar/$*.cpp src/simd/$*.cpp $(LDLIBS) -o $@

run: run-bench

run-bench: bench
	@for ex in $(EXERCISES); do echo "\n=== bench/$$ex ==="; ./$(BUILD_DIR)/$${ex}_bench; done

riscv: $(RISCV_TARGETS)

$(BUILD_DIR)/%_bench.riscv: bench/%.cpp src/scalar/%.cpp src/simd/%.cpp include/simd_examples/%.hpp bench/benchmark_common.hpp src/simd_common.h Makefile
	@mkdir -p $(BUILD_DIR)
	$(RISCV_CXX) $(RISCV_CXXFLAGS) bench/$*.cpp src/scalar/$*.cpp src/simd/$*.cpp $(RISCV_LDLIBS) -o $@

run-riscv-128: $(RISCV_TARGETS)
	@echo "=== Running RISC-V VLEN=128 ==="
	@for ex in $(EXERCISES); do \
		echo "\n=== $$ex (VLEN=128) ==="; \
		$(QEMU_RISCV) -cpu rv64,v=true,vlen=128 ./$(BUILD_DIR)/$${ex}_bench.riscv; \
	done

run-riscv-512: $(RISCV_TARGETS)
	@echo "=== Running RISC-V VLEN=512 ==="
	@for ex in $(EXERCISES); do \
		echo "\n=== $$ex (VLEN=512) ==="; \
		$(QEMU_RISCV) -cpu rv64,v=true,vlen=512 ./$(BUILD_DIR)/$${ex}_bench.riscv; \
	done

run-riscv-both: $(RISCV_TARGETS)
	@$(MAKE) run-riscv-128
	@echo "\n========================================\n"
	@$(MAKE) run-riscv-512

verify-riscv: $(BUILD_DIR)/01_add_simd.s
	@echo "=== Checking for RISC-V Vector (RVV) instructions ==="
	@grep -E 'vle32\.v|vse32\.v|vfadd\.vv|vfmacc\.vv' $< || echo "No RVV instructions found - check compiler flags"

$(BUILD_DIR)/01_add_simd.s: src/simd/01_add.cpp include/simd_examples/01_add.hpp src/simd_common.h Makefile
	@mkdir -p $(BUILD_DIR)
	$(RISCV_CXX) $(RISCV_CXXFLAGS) -S $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all bench run run-bench riscv run-riscv-128 run-riscv-512 run-riscv-both verify-riscv clean
