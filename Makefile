# Makefile for std::simd examples
# Cross-platform: Intel AVX-512 (MareNostrum 5), Apple Silicon (NEON), and RISC-V RVV
# Usage: make; make run; make riscv; make run-riscv-128

# Detect platform
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
    # macOS - requires Homebrew GCC for std::experimental::simd
    # Install with: brew install gcc
    CXX = g++-15
    ifeq ($(UNAME_M),arm64)
        # Apple Silicon (M1/M2/M3) - ARM NEON (128-bit vectors = 4 floats)
        CXXFLAGS = -std=c++2b -O3 -mcpu=apple-m1 -fno-math-errno -fno-trapping-math -Wall -Wextra -I.
    else
        # Intel Mac
        CXXFLAGS = -std=c++2b -O3 -march=native -fno-math-errno -fno-trapping-math -Wall -Wextra -I.
    endif
else
    # Linux (MareNostrum 5 / Intel AVX-512)
    CXX = g++
    CXXFLAGS = -std=c++2b -O3 -march=native -mavx512f -mavx512vl -mavx512dq -mavx512bw -fno-math-errno -fno-trapping-math -Wall -Wextra -I.
endif

# RISC-V cross-compilation settings
RISCV_CXX = riscv64-linux-gnu-g++
RISCV_CXXFLAGS = -std=c++2b -O3 -march=rv64gcv -static -fno-math-errno -fno-trapping-math -Wall -Wextra -I.
RISCV_LDLIBS = -lm
QEMU_RISCV = qemu-riscv64-static

LDLIBS = -lm
TARGETS = 01_add 02_sum 03_clamp 04_count 05_softmax 06_fma 07_filter 08_conv1d
TARGETS_RISCV = $(addsuffix .riscv,$(TARGETS))

all: $(TARGETS)

run: $(TARGETS)
	@for ex in $(TARGETS); do echo "\n=== $$ex ==="; ./$$ex; done

%: %.cpp common.h Makefile
	$(CXX) $(CXXFLAGS) $(LDLIBS) $< -o $@

# RISC-V targets
riscv: $(TARGETS_RISCV)

%.riscv: %.cpp common.h Makefile
	$(RISCV_CXX) $(RISCV_CXXFLAGS) $(RISCV_LDLIBS) $< -o $@

# Run RISC-V binaries with VLEN=128 (4 floats, like ARM NEON)
run-riscv-128: $(TARGETS_RISCV)
	@echo "=== Running RISC-V examples with VLEN=128 (4 floats per register) ==="
	@for ex in $(TARGETS); do \
		echo "\n=== $$ex (VLEN=128) ==="; \
		$(QEMU_RISCV) -cpu rv64,v=true,vlen=128 ./$$ex.riscv; \
	done

# Run RISC-V binaries with VLEN=512 (16 floats, like AVX-512)
run-riscv-512: $(TARGETS_RISCV)
	@echo "=== Running RISC-V examples with VLEN=512 (16 floats per register) ==="
	@for ex in $(TARGETS); do \
		echo "\n=== $$ex (VLEN=512) ==="; \
		$(QEMU_RISCV) -cpu rv64,v=true,vlen=512 ./$$ex.riscv; \
	done

# Run both VLEN configurations for comparison
run-riscv-both: $(TARGETS_RISCV)
	@$(MAKE) run-riscv-128
	@echo "\n========================================\n"
	@$(MAKE) run-riscv-512

# Generate assembly for RISC-V to verify vector instructions
verify-riscv: 01_add.s
	@echo "=== Checking for RISC-V Vector (RVV) instructions ==="
	@grep -E 'vle32\.v|vse32\.v|vfadd\.vv|vfmul\.vv|vfmacc\.vv' 01_add.s || echo "No RVV instructions found - check compiler flags"

01_add.s: 01_add.cpp common.h
	$(RISCV_CXX) $(RISCV_CXXFLAGS) -S $< -o $@

clean:
	rm -f $(TARGETS) $(TARGETS_RISCV) *.o *.s 06_conv 06_conv.riscv 08_conv 08_conv.riscv

.PHONY: all run riscv run-riscv-128 run-riscv-512 run-riscv-both verify-riscv clean
