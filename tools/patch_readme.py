import re

with open('README.md', 'r') as f:
    content = f.read()

# 1. Replace the entire "RISC-V Emulation" section up to "### Compiler Flags"
riscv_emulation_pattern = re.compile(
    r'#### RISC-V Emulation.*?### Compiler Flags',
    re.DOTALL
)

new_riscv_section = """#### RISC-V Emulation (MareNostrum 5 / Linux)

Modern compilers (like GCC 15.1.0) successfully lower `std::experimental::simd` to real RISC-V Vector (RVV) instructions natively, without needing third-party libraries or submodules.

**Note on Older Compilers**: If you use GCC 13 (the default on MareNostrum 5 and Ubuntu 24.04), `std::experimental::simd` will safely fall back to scalar code because RVV mapping wasn't implemented yet. 

To compile and emulate RVV on x86_64 machines (like MareNostrum 5), use a bleeding-edge cross-compiler and QEMU:

```bash
# 1. Download a bleeding-edge GCC 15.1.0 RISC-V cross-compiler (e.g., Bootlin toolchains)
wget https://toolchains.bootlin.com/downloads/releases/toolchains/riscv64-lp64d/tarballs/riscv64-lp64d--glibc--bleeding-edge-2025.08-1.tar.xz
tar xf riscv64-lp64d--glibc--bleeding-edge-2025.08-1.tar.xz
export PATH=$PWD/riscv64-lp64d--glibc--bleeding-edge-2025.08-1/bin:$PATH

# 2. Download a static QEMU user-mode emulator for RISC-V
wget https://github.com/multiarch/qemu-user-static/releases/download/v7.2.0-1/qemu-riscv64-static
chmod +x qemu-riscv64-static
export QEMU_RISCV=$PWD/qemu-riscv64-static

# 3. Build and verify RVV instruction generation
make riscv RISCV_CXX=riscv64-linux-g++ RISCV_CXXFLAGS="-std=c++2b -O3 -march=rv64gcv -mrvv-vector-bits=zvl -static"
make verify-riscv RISCV_CXX=riscv64-linux-g++ RISCV_OBJDUMP=riscv64-linux-objdump

# 4. Run through the QEMU emulator (e.g., with VLEN=128)
make run-riscv-128 QEMU_RISCV=$QEMU_RISCV
```

**Emulation Slowdown Notes:**
When running the binaries through QEMU (`vlen=128`), you will observe that the "Speedup" metrics are actually `< 1.0x` (meaning the SIMD code runs slower than the scalar code). This is completely expected: QEMU interprets and translates every RISC-V vector instruction into host x86 instructions dynamically in software. Vector execution via software emulation is incredibly expensive compared to native execution, so you will not see real performance gains in an emulator. Emulation here strictly serves to verify that the C++ code successfully compiles to RVV and computes logically correct results.

### Compiler Flags"""

content = riscv_emulation_pattern.sub(new_riscv_section, content)

# 2. Remove macOS compile-only line
content = re.sub(r'\| RISC-V \(macOS compile-only\).*?\n', '', content)

# 3. Update the bash block for RISC-V RVV (GCC 15.1)
old_bash = """**RISC-V RVV (GCC 15.1):**
```bash
riscv64-linux-g++ -std=c++2b -march=rv64gcv -O3 -Ibench -Iinclude -Isrc \\
  -DSIMD_EXAMPLES_SIMD bench/01_add.cpp src/simd/01_add.cpp \\
  -o /tmp/01_add_simd.riscv

riscv64-linux-objdump -d /tmp/01_add_simd.riscv | \\
  grep -E 'vsetvli|vle32\\.v|vse32\\.v|vfadd\\.vv|vfmul\\.vv|vfmacc\\.vv|vfred'
```"""

new_bash = """**RISC-V RVV (GCC 15.1):**
```bash
riscv64-linux-g++ -std=c++2b -march=rv64gcv -mrvv-vector-bits=zvl -O3 -Ibench -Iinclude -Isrc \\
  -DSIMD_EXAMPLES_SIMD bench/01_add.cpp src/simd/01_add.cpp \\
  -o /tmp/01_add_simd.riscv

riscv64-linux-objdump -d /tmp/01_add_simd.riscv | \\
  grep -E 'vsetvli|vle32\\.v|vse32\\.v|vfadd\\.vv|vfmul\\.vv|vfmacc\\.vv|vfred'
```"""
content = content.replace(old_bash, new_bash)

# 4. Update the text at the end of the table
old_text = "RISC-V support is compiler-version dependent. Modern GCC 15.1.0 cleanly emits RVV instructions in the binary. The older Linux/GCC 13 user-mode path will fall back to scalar `fadd.s` for `std::experimental::simd`. Execution of the GCC 15 RISC-V binaries may require setting up `qemu-riscv64` correctly on your OS."
new_text = "RISC-V support is compiler-version dependent. Modern GCC 15.1.0 cleanly emits RVV instructions natively. The older Linux/GCC 13 path will safely fall back to scalar operations like `fadd.s` for `std::experimental::simd`. As noted, executing these binaries on x86 machines requires `qemu-riscv64` and will exhibit emulation slowdowns."
content = content.replace(old_text, new_text)

with open('README.md', 'w') as f:
    f.write(content)

