# Scalar and std::simd algorithms with isolated benchmark executables.

BUILD_DIR := build
EXERCISES := 01_add 02_sum 03_clamp 04_count 05_softmax 06_fma 07_filter 08_conv1d
SCALAR_TARGETS := $(addprefix $(BUILD_DIR)/,$(addsuffix _scalar,$(EXERCISES)))
SIMD_TARGETS := $(addprefix $(BUILD_DIR)/,$(addsuffix _simd,$(EXERCISES)))
BENCH_TARGETS := $(SCALAR_TARGETS) $(SIMD_TARGETS)
RISCV_SCALAR_TARGETS := $(addprefix $(BUILD_DIR)/,$(addsuffix _scalar.riscv,$(EXERCISES)))
RISCV_SIMD_TARGETS := $(addprefix $(BUILD_DIR)/,$(addsuffix _simd.riscv,$(EXERCISES)))
RISCV_TARGETS := $(RISCV_SCALAR_TARGETS) $(RISCV_SIMD_TARGETS)
NATIVE_ASSEMBLY_TARGETS := \
    $(addprefix $(BUILD_DIR)/,$(addsuffix _scalar.s,$(EXERCISES))) \
    $(addprefix $(BUILD_DIR)/,$(addsuffix _simd.s,$(EXERCISES)))
BENCH_DEPS := bench/benchmark_common.hpp bench/benchmark_implementation.hpp \
    bench/benchmark_reference.hpp Makefile

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

scalar: $(SCALAR_TARGETS)

simd: $(SIMD_TARGETS)

$(BUILD_DIR)/%_scalar: bench/%.cpp src/scalar/%.cpp include/simd_examples/%.hpp $(BENCH_DEPS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DSIMD_EXAMPLES_SCALAR bench/$*.cpp src/scalar/$*.cpp $(LDLIBS) -o $@

$(BUILD_DIR)/%_simd: bench/%.cpp src/simd/%.cpp include/simd_examples/%.hpp $(BENCH_DEPS) src/simd_common.h
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DSIMD_EXAMPLES_SIMD bench/$*.cpp src/simd/$*.cpp $(LDLIBS) -o $@

run: run-bench

run-bench: bench
	@for ex in $(EXERCISES); do \
		echo "\n=== bench/$$ex (scalar) ==="; ./$(BUILD_DIR)/$${ex}_scalar; \
		echo "\n=== bench/$$ex (simd) ==="; ./$(BUILD_DIR)/$${ex}_simd; \
	done

assembly: $(NATIVE_ASSEMBLY_TARGETS)

$(BUILD_DIR)/%_scalar.s: src/scalar/%.cpp include/simd_examples/%.hpp Makefile
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DSIMD_EXAMPLES_SCALAR -S $< -o $@

$(BUILD_DIR)/%_simd.s: src/simd/%.cpp include/simd_examples/%.hpp src/simd_common.h Makefile
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DSIMD_EXAMPLES_SIMD -S $< -o $@

riscv: $(RISCV_TARGETS)

$(BUILD_DIR)/%_scalar.riscv: bench/%.cpp src/scalar/%.cpp include/simd_examples/%.hpp $(BENCH_DEPS)
	@mkdir -p $(BUILD_DIR)
	$(RISCV_CXX) $(RISCV_CXXFLAGS) -DSIMD_EXAMPLES_SCALAR bench/$*.cpp src/scalar/$*.cpp $(RISCV_LDLIBS) -o $@

$(BUILD_DIR)/%_simd.riscv: bench/%.cpp src/simd/%.cpp include/simd_examples/%.hpp $(BENCH_DEPS) src/simd_common.h
	@mkdir -p $(BUILD_DIR)
	$(RISCV_CXX) $(RISCV_CXXFLAGS) -DSIMD_EXAMPLES_SIMD bench/$*.cpp src/simd/$*.cpp $(RISCV_LDLIBS) -o $@

run-riscv-128: $(RISCV_TARGETS)
	@echo "=== Running RISC-V VLEN=128 ==="
	@for ex in $(EXERCISES); do \
		echo "\n=== $$ex scalar (VLEN=128) ==="; \
		$(QEMU_RISCV) -cpu rv64,v=true,vlen=128 ./$(BUILD_DIR)/$${ex}_scalar.riscv; \
		echo "\n=== $$ex simd (VLEN=128) ==="; \
		$(QEMU_RISCV) -cpu rv64,v=true,vlen=128 ./$(BUILD_DIR)/$${ex}_simd.riscv; \
	done

run-riscv-512: $(RISCV_TARGETS)
	@echo "=== Running RISC-V VLEN=512 ==="
	@for ex in $(EXERCISES); do \
		echo "\n=== $$ex scalar (VLEN=512) ==="; \
		$(QEMU_RISCV) -cpu rv64,v=true,vlen=512 ./$(BUILD_DIR)/$${ex}_scalar.riscv; \
		echo "\n=== $$ex simd (VLEN=512) ==="; \
		$(QEMU_RISCV) -cpu rv64,v=true,vlen=512 ./$(BUILD_DIR)/$${ex}_simd.riscv; \
	done

run-riscv-both: $(RISCV_TARGETS)
	@$(MAKE) run-riscv-128
	@echo "\n========================================\n"
	@$(MAKE) run-riscv-512

verify-riscv: $(BUILD_DIR)/01_add_simd.riscv.s
	@echo "=== Checking for RISC-V Vector (RVV) instructions ==="
	@grep -E 'vle32\.v|vse32\.v|vfadd\.vv|vfmacc\.vv' $< || echo "No RVV instructions found - check compiler flags"

$(BUILD_DIR)/01_add_simd.riscv.s: src/simd/01_add.cpp include/simd_examples/01_add.hpp src/simd_common.h Makefile
	@mkdir -p $(BUILD_DIR)
	$(RISCV_CXX) $(RISCV_CXXFLAGS) -DSIMD_EXAMPLES_SIMD -S $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all bench scalar simd run run-bench assembly riscv run-riscv-128 run-riscv-512 \
    run-riscv-both verify-riscv clean
