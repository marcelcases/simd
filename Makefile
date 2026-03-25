# Makefile for std::simd examples
# Cross-platform: Intel AVX-512 (MareNostrum 5) and Apple Silicon (NEON)
# Usage: make; make run

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

LDLIBS = -lm
TARGETS = 01_add 02_sum 03_clamp 04_count 05_softmax 06_conv 07_filter

all: $(TARGETS)

run: $(TARGETS)
	@for ex in $(TARGETS); do echo "\n=== $$ex ==="; ./$$ex; done

%: %.cpp common.h Makefile
	$(CXX) $(CXXFLAGS) $(LDLIBS) $< -o $@

clean:
	rm -f $(TARGETS) *.o

.PHONY: all run clean
