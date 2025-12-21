"""
Performance and scalability utilities for WebCFD_AI pipeline.
Provides parallel processing, memory optimization, and progress tracking.
"""

import os
import time
import psutil
import logging
from typing import Any, Callable, List, Optional, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import wraps
import multiprocessing as mp
import numpy as np
import pyvista as pv
from tqdm import tqdm
import gc

logger = logging.getLogger(__name__)

# Performance configuration
MAX_WORKERS = min(mp.cpu_count(), 8)  # Limit to prevent memory issues
MEMORY_THRESHOLD = 0.8  # 80% memory usage threshold
CHUNK_SIZE = 1000  # Default chunk size for processing


class PerformanceMonitor:
    """Monitor and track performance metrics during pipeline execution."""
    
    def __init__(self):
        self.start_time = None
        self.memory_usage = []
        self.peak_memory = 0
        self.operation_times = {}
        
    def start(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.memory_usage = []
        self.peak_memory = 0
        
    def update_memory(self):
        """Update current memory usage."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(memory_mb)
        self.peak_memory = max(self.peak_memory, memory_mb)
        
        # Check if memory usage is too high
        if memory_mb > psutil.virtual_memory().total * MEMORY_THRESHOLD / 1024 / 1024:
            logger.warning(f"High memory usage detected: {memory_mb:.1f} MB")
            gc.collect()  # Force garbage collection
            
    def log_operation(self, operation_name: str, duration: float):
        """Log operation timing."""
        self.operation_times[operation_name] = duration
        logger.info(f"Operation '{operation_name}' completed in {duration:.2f}s")
        
    def get_summary(self) -> dict:
        """Get performance summary."""
        total_time = time.time() - self.start_time if self.start_time else 0
        return {
            'total_time': total_time,
            'peak_memory_mb': self.peak_memory,
            'avg_memory_mb': np.mean(self.memory_usage) if self.memory_usage else 0,
            'operation_times': self.operation_times
        }


def memory_efficient_processing(func: Callable) -> Callable:
    """Decorator for memory-efficient processing with automatic cleanup."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = PerformanceMonitor()
        monitor.start()
        
        try:
            result = func(*args, **kwargs)
            monitor.update_memory()
            return result
        finally:
            # Force cleanup
            gc.collect()
            monitor.update_memory()
            
    return wrapper


def parallel_process(
    items: List[Any], 
    func: Callable, 
    max_workers: Optional[int] = None,
    use_threads: bool = False,
    chunk_size: Optional[int] = None,
    desc: str = "Processing"
) -> List[Any]:
    """
    Process items in parallel with progress tracking and memory management.
    
    Args:
        items: List of items to process
        func: Function to apply to each item
        max_workers: Maximum number of workers (default: MAX_WORKERS)
        use_threads: Use ThreadPoolExecutor instead of ProcessPoolExecutor
        chunk_size: Size of chunks for processing
        desc: Description for progress bar
        
    Returns:
        List of processed results
    """
    if not items:
        return []
        
    max_workers = max_workers or MAX_WORKERS
    chunk_size = chunk_size or CHUNK_SIZE
    
    # For small datasets, use single-threaded processing
    if len(items) < max_workers * 2:
        logger.info(f"Small dataset ({len(items)} items), using single-threaded processing")
        return [func(item) for item in tqdm(items, desc=desc)]
    
    # Chunk items for better memory management
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    results = []
    with executor_class(max_workers=max_workers) as executor:
        # Submit all chunks
        future_to_chunk = {
            executor.submit(process_chunk, chunk, func): chunk 
            for chunk in chunks
        }
        
        # Process completed chunks with progress bar
        for future in tqdm(as_completed(future_to_chunk), total=len(chunks), desc=desc):
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                # Continue with other chunks
                
    return results


def process_chunk(chunk: List[Any], func: Callable) -> List[Any]:
    """Process a chunk of items."""
    return [func(item) for item in chunk]


def optimize_mesh_memory(mesh: pv.PolyData) -> pv.PolyData:
    """
    Optimize mesh memory usage by removing unnecessary data and compressing.
    
    Args:
        mesh: PyVista mesh to optimize
        
    Returns:
        Optimized mesh
    """
    # Remove unused points
    mesh = mesh.clean()
    
    # Remove duplicate points - use the correct method for single mesh
    mesh = mesh.merge_points(tolerance=1e-6)
    
    # Optimize data types
    if 'patch_id' in mesh.cell_data:
        # Use smallest integer type that can hold the data
        patch_ids = mesh.cell_data['patch_id']
        if patch_ids.max() < 255:
            mesh.cell_data['patch_id'] = patch_ids.astype(np.uint8)
        elif patch_ids.max() < 65535:
            mesh.cell_data['patch_id'] = patch_ids.astype(np.uint16)
        else:
            mesh.cell_data['patch_id'] = patch_ids.astype(np.uint32)
    
    return mesh


def estimate_memory_usage(mesh: pv.PolyData) -> float:
    """Estimate memory usage of a mesh in MB."""
    points_memory = mesh.points.nbytes / 1024 / 1024
    faces_memory = mesh.faces.nbytes / 1024 / 1024
    cell_data_memory = sum(
        data.nbytes for data in mesh.cell_data.values()
    ) / 1024 / 1024
    point_data_memory = sum(
        data.nbytes for data in mesh.point_data.values()
    ) / 1024 / 1024
    
    return points_memory + faces_memory + cell_data_memory + point_data_memory


def create_progress_callback(total: int, desc: str = "Processing"):
    """Create a progress callback for long-running operations."""
    pbar = tqdm(total=total, desc=desc, unit="items")
    
    def callback(completed: int = 1):
        pbar.update(completed)
        
    def close():
        pbar.close()
        
    return callback, close


def log_performance(func: Callable) -> Callable:
    """Decorator to log performance metrics for functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            logger.info(
                f"{func.__name__}: {duration:.2f}s, "
                f"memory: {start_memory:.1f}MB -> {end_memory:.1f}MB "
                f"(Δ{memory_delta:+.1f}MB)"
            )
            
    return wrapper
