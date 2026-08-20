# 1. Get an interactive node (or put this in your Slurm script)
# salloc --nodes=1 --tasks-per-node=1 --cpus-per-task=128 --time=01:00:00

# 2. Load the compiler environment
module purge
module load intel/2025.2
module load gcc/14.1.0_binutils241

# 3. Verify the compiler
# g++ --version

# 4. Build and run explicit SIMD examples
# make simd
# ./build/01_add_simd
