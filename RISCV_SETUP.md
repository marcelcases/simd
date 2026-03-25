# RISC-V Emulator Setup for SIMD Testing

This guide explains how to set up a RISC-V emulator with vector extensions (RVV) to test the `std::simd` examples.

## Option 1: QEMU User-Mode (Recommended for Quick Testing)

QEMU user-mode emulation allows running RISC-V binaries directly without booting a full system.

### Step 1: Install QEMU with RISC-V support

```bash
# macOS
brew install qemu

# Verify QEMU has RISC-V support
qemu-riscv64 --version
```

### Step 2: Install RISC-V toolchain

```bash
# macOS - Install RISC-V GNU toolchain
brew tap riscv-software-src/riscv
brew install riscv-gnu-toolchain

# Verify installation
riscv64-unknown-elf-gcc --version
```

Alternatively, use a pre-built toolchain:

```bash
# Download from https://github.com/riscv-collab/riscv-gnu-toolchain/releases
# Or use the SiFive toolchain
wget https://static.dev.sifive.com/dev-tools/freedom-tools/v2020.12/riscv64-unknown-elf-toolchain-10.2.0-2020.12.8-x86_64-apple-darwin.tar.gz
tar xzf riscv64-unknown-elf-toolchain-*.tar.gz
export PATH=$PWD/riscv64-unknown-elf-toolchain-*/bin:$PATH
```

### Step 3: Cross-compile examples for RISC-V

Create a RISC-V specific Makefile or modify the existing one:

```makefile
# Makefile.riscv
CXX = riscv64-unknown-linux-gnu-g++
CXXFLAGS = -std=c++2b -O3 -march=rv64gcv -static
CXXFLAGS += -fno-math-errno -fno-trapping-math
CXXFLAGS += -Wall -Wextra -I.

TARGETS = 01_add 02_sum 03_clamp 04_count 05_softmax 06_conv 07_filter

all: $(TARGETS)

%: %.cpp common.h
	$(CXX) $(CXXFLAGS) $< -o $@.riscv -lm

clean:
	rm -f *.riscv
```

Key flags:
- `-march=rv64gcv`: RV64 with vector extension (V)
- `-static`: Statically link to avoid library compatibility issues with QEMU

### Step 4: Run with QEMU

```bash
# Compile
make -f Makefile.riscv 01_add

# Run with QEMU user-mode
qemu-riscv64 -cpu rv64,v=true,vlen=128 ./01_add.riscv
```

QEMU flags:
- `-cpu rv64,v=true`: Enable vector extension
- `vlen=128`: Set vector register length (128, 256, 512, or 1024 bits)

## Option 2: Spike (RISC-V ISA Simulator)

Spike is the official RISC-V ISA simulator with full RVV support.

### Installation

```bash
# Clone and build Spike
git clone https://github.com/riscv-software-src/riscv-isa-sim.git
cd riscv-isa-sim
mkdir build && cd build
../configure --prefix=$HOME/.local
make -j$(nproc)
make install

# Clone proxy kernel (needed to run programs)
git clone https://github.com/riscv-software-src/riscv-pk.git
cd riscv-pk
mkdir build && cd build
../configure --prefix=$HOME/.local --host=riscv64-unknown-elf
make -j$(nproc)
make install
```

### Run with Spike

```bash
# Compile statically linked binary
riscv64-unknown-elf-g++ -std=c++2b -O3 -march=rv64gcv -static 01_add.cpp -o 01_add.riscv

# Run with Spike
spike --isa=RV64GCV pk 01_add.riscv
```

## Option 3: Docker Container (Easiest Setup)

Use a pre-configured Docker container with RISC-V toolchain and QEMU.

```bash
# Pull RISC-V development image
docker pull riscv64/ubuntu:22.04

# Run container with project mounted
docker run -it -v $(pwd):/work riscv64/ubuntu:22.04 bash

# Inside container:
cd /work
apt-get update && apt-get install -y g++ make
g++ -std=c++2b -O3 -march=rv64gcv 01_add.cpp -o 01_add
./01_add
```

## Option 4: Using Conda Environment

Install QEMU and cross-compilation tools via conda:

```bash
# Create new conda environment for RISC-V
conda create -n riscv python=3.11
conda activate riscv

# Install QEMU (if available in conda-forge)
conda install -c conda-forge qemu

# For toolchain, you'll still need to download manually or use Docker
# as native RISC-V GCC is not readily available in conda
```

## Verifying RISC-V Vector Instructions

After compilation, verify RVV instructions are generated:

```bash
# Generate assembly
riscv64-unknown-linux-gnu-g++ -std=c++2b -O3 -march=rv64gcv -S 01_add.cpp -o 01_add.s

# Look for vector instructions
grep -E 'vle|vse|vfadd|vfmul' 01_add.s

# Example RVV instructions:
# vle32.v    - vector load (32-bit elements)
# vse32.v    - vector store
# vfadd.vv   - vector float add
# vfmul.vv   - vector float multiply
# vfmacc.vv  - vector fused multiply-accumulate
```

## Expected Behavior

With RVV vector length = 128 bits (VLEN=128):
- `native_simd<float>::size()` should return 4 (128 bits / 32 bits per float)
- Similar to ARM NEON behavior

With VLEN=512:
- Should return 16 (like AVX-512)

## Troubleshooting

### Issue: `std::experimental::simd` not found

RISC-V support in GCC for `std::experimental::simd` requires GCC 13+:

```bash
riscv64-unknown-linux-gnu-g++ --version
# Should be >= 13.0
```

If not available, you may need to build GCC from source:

```bash
git clone https://github.com/riscv-collab/riscv-gnu-toolchain
cd riscv-gnu-toolchain
./configure --prefix=$HOME/.local/riscv --with-arch=rv64gcv
make linux -j$(nproc)
```

### Issue: QEMU doesn't recognize vector instructions

Update to latest QEMU (5.1+):

```bash
brew upgrade qemu  # macOS
```

### Issue: Illegal instruction errors

Ensure QEMU is invoked with vector extension enabled:

```bash
qemu-riscv64 -cpu rv64,v=true,vlen=128 ./program.riscv
```

## Recommended Approach

For quickest setup on macOS:

1. **Install QEMU via Homebrew**: `brew install qemu`
2. **Use Docker for cross-compilation**: Pre-built toolchain in container
3. **Mount your repo**: Run binaries with QEMU

Example workflow:

```bash
# Terminal 1: Build in Docker
docker run -it -v $(pwd):/work --rm riscv64/ubuntu:22.04 bash
cd /work
apt-get update && apt-get install -y g++-12
g++-12 -std=c++2b -O3 -march=rv64gcv -static 01_add.cpp -o 01_add.riscv
exit

# Terminal 2: Run with local QEMU
qemu-riscv64 -cpu rv64,v=true,vlen=128 ./01_add.riscv
```

## Adding Results to README

After running benchmarks, add RISC-V column to the benchmark table:

```markdown
| Example | MN5 Speedup | M1 Pro Speedup | RV64V Speedup |
|---------|-------------|----------------|---------------|
| 01_add  | 1.2x        | 1.2x           | ?             |
```

And to the verification table:

```markdown
| Example | ARM NEON | x86 AVX-512 | RISC-V RVV |
|---------|----------|-------------|------------|
| 01_add  | fadd v.4s | vaddps zmm | vfadd.vv   |
```
