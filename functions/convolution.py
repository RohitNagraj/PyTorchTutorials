import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.kernels = torch.randn((out_channels, in_channels, kernel_size, kernel_size))
    
    def forward(self, input: torch.Tensor, kernels=None):
        """
        Optimized using unfold (im2col approach) for vectorized computation
        """
        in_channels, input_h, input_w = input.shape
        
        if kernels is None:
            kernels = self.kernels
        
        # Use unfold to extract all patches at once (im2col approach)
        # unfold(dimension, size, step)
        # First unfold along height (dim=1), then width (dim=2)
        # Shape after first unfold: (in_channels, output_h, input_w, kernel_size)
        # Shape after second unfold: (in_channels, output_h, output_w, kernel_size, kernel_size)
        patches = input.unfold(1, self.kernel_size, 1).unfold(2, self.kernel_size, 1) # Shape: (3, 62, 62, 3, 3)
        output = torch.einsum("ihwkj,oikj->ohw", patches, kernels)
        
        return output


# Alternative: Using F.conv2d (fastest, leverages optimized backends)
class Conv2dBuiltin(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.kernels = torch.randn((out_channels, in_channels, kernel_size, kernel_size))
    
    def forward(self, input: torch.Tensor, kernels=None):
        if kernels is None:
            kernels = self.kernels
        
        # Add batch dimension for F.conv2d
        input_batched = input.unsqueeze(0)
        output = F.conv2d(input_batched, kernels, padding=0)
        
        # Remove batch dimension
        return output.squeeze(0)


# Benchmark comparison
if __name__ == "__main__":
    import time
    
    # Test parameters
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    input_size = 64
    
    # Create test input
    input_tensor = torch.randn(in_channels, input_size, input_size)
    
    # Original implementation (your code)
    class Conv2dOriginal(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.kernels = torch.randn((out_channels, in_channels, kernel_size, kernel_size))
        
        def forward(self, input: torch.Tensor, kernels=None):
            in_channels, input_h, input_w = input.shape
            if kernels is None:
                kernels = self.kernels
            output_shape = (self.out_channels, input_h-self.kernel_size+1, input_w-self.kernel_size+1)
            output = torch.empty(output_shape)
            for out_channel in range(output_shape[0]):
                for h in range(output_shape[1]):
                    for w in range(output_shape[2]):
                        patch = input[:,h:h+self.kernel_size, w:w+self.kernel_size]
                        o = (patch * kernels[out_channel]).sum()
                        output[out_channel,h,w] = o
            return output
    
    # Initialize models with same kernels
    kernels = torch.randn((out_channels, in_channels, kernel_size, kernel_size))
    
    original = Conv2dOriginal(in_channels, out_channels, kernel_size)
    original.kernels = kernels.clone()
    
    optimized = Conv2d(in_channels, out_channels, kernel_size)
    optimized.kernels = kernels.clone()
    
    builtin = Conv2dBuiltin(in_channels, out_channels, kernel_size)
    builtin.kernels = kernels.clone()
    
    # Warmup
    _ = original(input_tensor)
    _ = optimized(input_tensor)
    _ = builtin(input_tensor)
    
    # Benchmark original
    n_runs = 10
    start = time.time()
    for _ in range(n_runs):
        out1 = original(input_tensor)
    time_original = (time.time() - start) / n_runs
    
    # Benchmark optimized
    start = time.time()
    for _ in range(n_runs):
        out2 = optimized(input_tensor)
    time_optimized = (time.time() - start) / n_runs
    
    # Benchmark builtin
    start = time.time()
    for _ in range(n_runs):
        out3 = builtin(input_tensor)
    time_builtin = (time.time() - start) / n_runs
    
    # Verify correctness
    print("Output shapes match:", out1.shape == out2.shape == out3.shape)
    print("Optimized vs Original max diff:", (out1 - out2).abs().max().item())
    print("Builtin vs Original max diff:", (out1 - out3).abs().max().item())
    print()
    print(f"Original time: {time_original*1000:.2f}ms")
    print(f"Optimized time: {time_optimized*1000:.2f}ms")
    print(f"Builtin time: {time_builtin*1000:.2f}ms")
    print(f"\nSpeedup (Optimized): {time_original/time_optimized:.1f}x")
    print(f"Speedup (Builtin): {time_original/time_builtin:.1f}x")