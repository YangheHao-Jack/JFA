import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from cucim.core.operations.morphology import distance_transform_edt

# Device setup
device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize grid with seeds for 2D
def initialize_seed_grid_2D(shape, seed_positions, device):
    grid = torch.full(shape, float('inf'), device=device)
    for pos in seed_positions:
        grid[pos] = 0
    return grid

# Initialize grid with seeds for 3D
def initialize_seed_grid_3D(shape, seed_positions, device):
    grid = torch.full(shape, float('inf'), device=device)
    for pos in seed_positions:
        grid[pos] = 0
    return grid

import torch.nn.functional as F

# 2D Jump Flood Algorithm using max-pooling
def jump_flood_2D(grid, max_iter=100, device=device_cpu, debug=False):

    device = grid.device
    height, width = grid.shape
    jump = max(height, width) // 2  # Start with the maximum jump size

    # Initialize the output grid with the seed grid
    output_grid = grid.clone()

    for step in range(max_iter):
        kernel_size = 3
        dilation = jump
        effective_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
        padding = (effective_kernel_size - 1) // 2  # Calculate padding

        # Apply manual padding to the grid
        padded_grid = F.pad(
            output_grid.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
            (padding, padding, padding, padding),  # Pad left, right, top, bottom
            mode='replicate'  # Replicate edge values
        )

        # Perform max pooling with dilation and no internal padding
        dilated_grid = F.max_pool2d(
            -padded_grid,  # Negate values for max pooling to mimic min behavior
            kernel_size=kernel_size,
            stride=1,
            padding=0,  # Padding is already handled manually
            dilation=dilation
        )
        dilated_grid = -dilated_grid.squeeze(0).squeeze(0)  # Undo the negation and remove extra dimensions

        # Combine new seeds with previously accumulated clusters
        propagated_seeds = dilated_grid + jump  # Add jump size for propagation
        output_grid = torch.min(output_grid, propagated_seeds)  # Keep the smallest (closest seed)

        # Debug: Print dynamically updated seed positions
        if debug:
            current_seed_positions = torch.nonzero(output_grid != float('inf'), as_tuple=False).tolist()
            print(f"Iteration {step + 1}, Jump Size {jump}: Seeds at {current_seed_positions}")

        # Halve the jump size for the next iteration
        jump = max(jump // 2, 1)

    return output_grid
# 3D Jump Flood Algorithm using max-pooling
def jump_flood_3D(grid, max_iter=5, device=device_cpu, debug=False):
    device = grid.device
    depth, height, width = grid.shape
    jump = max(depth, height, width) // 2  # Start with the maximum jump size

    # Initialize the output grid with the seed grid
    output_grid = grid.clone()

    for step in range(max_iter):
        kernel_size = 3
        dilation = jump
        effective_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
        padding = (effective_kernel_size - 1) // 2  # Calculate padding

        # Apply manual padding to the grid
        padded_grid = F.pad(
            output_grid.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
            (padding, padding, padding, padding, padding, padding),  # Pad depth, height, width
            mode='replicate'  # Replicate edge values
        )

        # Perform max pooling with dilation and no internal padding
        dilated_grid = F.max_pool3d(
            -padded_grid,  # Negate values for max pooling to mimic min behavior
            kernel_size=kernel_size,
            stride=1,
            padding=0,  # Padding is already handled manually
            dilation=dilation
        )
        dilated_grid = -dilated_grid.squeeze(0).squeeze(0)  # Undo the negation and remove extra dimensions

        # Combine new seeds with previously accumulated clusters
        propagated_seeds = dilated_grid + jump  # Add jump size for propagation
        output_grid = torch.min(output_grid, propagated_seeds)  # Keep the smallest (closest seed)

        # Debug: Print dynamically updated seed positions
        if debug:
            current_seed_positions = torch.nonzero(output_grid != float('inf'), as_tuple=False).tolist()
            print(f"Iteration {step + 1}, Jump Size {jump}: Seeds at {current_seed_positions}")

        # Halve the jump size for the next iteration
        jump = max(jump // 2, 1)

    return output_grid
# Benchmarking 2D JFA and cuCIM
def benchmark_jfa_2d_with_cucim(shape, seed_positions, max_iter=10):
    # Initialize seed grids for CPU and GPU
    grid_cpu = initialize_seed_grid_2D(shape, seed_positions, device_cpu)
    grid_gpu = initialize_seed_grid_2D(shape, seed_positions, device_gpu)
    
    # JFA on CPU
    start = time.time()
    distance_map_cpu = jump_flood_2D(grid_cpu, max_iter=max_iter, device=device_cpu)
    cpu_time = time.time() - start
    
    # JFA on GPU
    start = time.time()
    if device_gpu == torch.device("cuda"):
        torch.cuda.synchronize()
    distance_map_gpu = jump_flood_2D(grid_gpu, max_iter=max_iter, device=device_gpu)
    if device_gpu == torch.device("cuda"):
        torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    # cuCIM Distance Transform on GPU with initial grid of cp.inf
    cucim_init_grid = cp.full(shape, cp.inf, dtype=cp.float64)
    for pos in seed_positions:
        cucim_init_grid[pos] = 0  # Set seed positions to 0
    
    start = time.time()
    cucim_distance_map = distance_transform_edt(cucim_init_grid, float64_distances=True)
    cucim_time = time.time() - start
    
    # Return the execution times
    return cpu_time, gpu_time, cucim_time

# Benchmarking 3D JFA and cuCIM
def benchmark_jfa_3d_with_cucim(shape, seed_positions, max_iter=5):
    # Initialize seed grids for CPU and GPU
    grid_cpu = initialize_seed_grid_3D(shape, seed_positions, device_cpu)
    grid_gpu = initialize_seed_grid_3D(shape, seed_positions, device_gpu)
    
    # JFA on CPU
    start = time.time()
    distance_map_cpu = jump_flood_3D(grid_cpu, max_iter=max_iter, device=device_cpu)
    cpu_time = time.time() - start
    
    # JFA on GPU
    start = time.time()
    if device_gpu == torch.device("cuda"):
        torch.cuda.synchronize()
    distance_map_gpu = jump_flood_3D(grid_gpu, max_iter=max_iter, device=device_gpu)
    if device_gpu == torch.device("cuda"):
        torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    # cuCIM Distance Transform on GPU with initial grid of cp.inf
    cucim_init_grid = cp.full(shape, cp.inf, dtype=cp.float64)
    for pos in seed_positions:
        cucim_init_grid[pos] = 0  # Set seed positions to 0
    
    start = time.time()
    cucim_distance_map = distance_transform_edt(cucim_init_grid, float64_distances=True)
    cucim_time = time.time() - start
    
    # Return the execution times
    return cpu_time, gpu_time, cucim_time

# Define spatial sizes and run benchmarks for 2D and 3D
spatial_sizes_2D = [64, 128, 256, 512, 1024]  # 2D spatial sizes
spatial_sizes_3D = [64, 128, 256, 512]        # 3D spatial sizes

# Example seed positions
seed_positions_2D = [(32, 32), (16, 48), (48, 16), (32, 48)]
seed_positions_3D = [(16, 16, 16), (8, 32, 24), (24, 8, 32), (16, 24, 8)]

# Store results for 2D and 3D
cpu_times_2D, gpu_times_2D, cucim_times_2D = [], [], []
cpu_times_3D, gpu_times_3D, cucim_times_3D = [], [], []

# Run benchmarks for 2D sizes
for size in spatial_sizes_2D:
    shape = (size, size)
    cpu_time, gpu_time, cucim_time = benchmark_jfa_2d_with_cucim(shape, seed_positions_2D, max_iter=10)
    cpu_times_2D.append(cpu_time)
    gpu_times_2D.append(gpu_time)
    cucim_times_2D.append(cucim_time)
    print(f"2D {size}x{size}: CPU = {cpu_time:.4f}s, GPU = {gpu_time:.4f}s, cuCIM = {cucim_time:.4f}s")

# Run benchmarks for 3D sizes
for size in spatial_sizes_3D:
    shape = (size, size, size)
    cpu_time, gpu_time, cucim_time = benchmark_jfa_3d_with_cucim(shape, seed_positions_3D, max_iter=5)
    cpu_times_3D.append(cpu_time)
    gpu_times_3D.append(gpu_time)
    cucim_times_3D.append(cucim_time)
    print(f"3D {size}x{size}x{size}: CPU = {cpu_time:.4f}s, GPU = {gpu_time:.4f}s, cuCIM = {cucim_time:.4f}s")

# Plotting the execution times for 2D and 3D
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Execution Time vs Spatial Size for Different Methods (2D and 3D)")

# Plot for 2D
axes[0].plot(spatial_sizes_2D, cpu_times_2D, 'o-', color='red', label="CPU JFA")
axes[0].plot(spatial_sizes_2D, gpu_times_2D, 'o-', color='purple', label="GPU JFA")
axes[0].plot(spatial_sizes_2D, cucim_times_2D, 'o-', color='green', label="cuCIM")
axes[0].set_xlabel("Spatial Size (2D)")
axes[0].set_ylabel("Execution Time (seconds)")
axes[0].set_title("2D images")
  # Set x-axis to log scale
axes[0].set_yscale("log") 
axes[0].legend()
axes[0].grid()

# Plot for 3D
axes[1].plot(spatial_sizes_3D, cpu_times_3D, 'o-', color='red', label="CPU JFA")
axes[1].plot(spatial_sizes_3D, gpu_times_3D, 'o-', color='purple', label="GPU JFA")
axes[1].plot(spatial_sizes_3D, cucim_times_3D, 'o-', color='green', label="cuCIM")
axes[1].set_xlabel("Spatial Size (3D)")
axes[1].set_ylabel("Execution Time (seconds)")
axes[1].set_title("3D volumes")
  # Set x-axis to log scale
axes[1].set_yscale("log") 
axes[1].legend()
axes[1].grid()

plt.show()
