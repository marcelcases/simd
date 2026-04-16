
# 1. Get an interactive node (or put this in your Slurm script)
# salloc --nodes=1 --tasks-per-node=1 --cpus-per-task=128 --time=01:00:00

# 2. Load Intel oneAPI (gives recent GCC with full <experimental/simd>)
module purge
module load intel/2025.2     # or intel/2025.x when available
module load gcc/13.2.0        # or newer version that appears

# 3. Verify GCC version (must be >= 11, preferably 13+)
g++ --version

# 4. Compile
# make

# 5. Run
# ./simd_add_bench