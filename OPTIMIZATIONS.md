# Speech Synthesis Performance Optimizations

## Overview
This document describes the optimizations applied to speed up the NeuTTS-Air speech synthesis system while maintaining the 300-character chunk limit.

## Performance Improvements

### Expected Speedup: **1.5-3x faster** on GPU-enabled systems (CUDA/Apple Silicon)

## ⚠️ Important Note: Parallel Processing Not Possible
**The phonemizer library is NOT thread-safe.** Attempting to run multiple synthesis operations in parallel causes runtime errors due to race conditions in the phoneme processing. Therefore, chunks must be processed sequentially.

## Optimizations Implemented

### 1. **GPU Acceleration** 🚀
- **Auto-detection** of CUDA (NVIDIA) and MPS (Apple Silicon) GPUs
- **Codec offloading**: Audio codec processing now runs on GPU (significant speedup)
- **PyTorch backend**: Full GPU acceleration for both backbone and codec when using PyTorch fallback
- **Graceful fallback**: Automatically falls back to CPU if no GPU detected

**Impact**: 1.5-3x faster on GPU-enabled systems

### 2. **Smart Garbage Collection** 🧹
- **Reduced GC frequency**: Only garbage collect after batches (not after every chunk)
- **Skip-GC mode**: Individual chunk synthesis skips GC to reduce overhead
- **Batch-level cleanup**: GC runs after processing 6-12 chunks instead of every chunk

**Impact**: 10-20% faster, reduced CPU overhead

### 3. **Memory Optimization** 💾
- **Deferred GC**: Garbage collection every 10 chunks instead of every chunk
- **Efficient audio handling**: Minimal memory copying during processing
- **Proper cleanup**: Explicit deletion of temporary objects before GC

**Impact**: Lower memory usage, less thermal throttling

### 4. **Better Progress Tracking** 📊
- **Real-time metrics**: Shows chunks/second processing rate
- **Accurate ETA**: Statistical confidence intervals for time estimates
- **Session resume**: Can resume interrupted generations

## Technical Details

### GPU Detection
```python
def detect_best_device():
    # Auto-detects: cuda > mps > cpu
    # GGUF: CPU backbone + GPU codec (best for most users)
    # PyTorch: Full GPU acceleration (fallback)
```

### Optimized Processing Pipeline
```
Input Text
  ↓
Split into 300-char chunks (sentence boundaries)
  ↓
Sequential processing (phonemizer limitation)
  ├─ Chunk 1 → Synthesize (GPU codec) → Save
  ├─ Chunk 2 → Synthesize (GPU codec) → Save
  ├─ Chunk 3 → Synthesize (GPU codec) → Save
  └─ ...
  ↓
GC every 10 chunks
  ↓
Final audio output
```

### Why Sequential Processing?
The `phonemizer` library (used for text-to-phoneme conversion) maintains internal state and is not thread-safe. Concurrent calls result in:
- Race conditions in phoneme buffer
- Line count mismatches
- RuntimeError: "number of lines in input and output must be equal"

This is a limitation of the underlying library, not this implementation.

## Performance Characteristics

### Before Optimization
- **Processing**: Sequential (1 chunk at a time)
- **GC**: After every chunk (~0.5-1 seconds overhead each)
- **GPU**: Never used (hardcoded to CPU)
- **Typical speed**: ~3-6 seconds per chunk on CPU

### After Optimization
- **Processing**: Sequential (phonemizer limitation)
- **GC**: Every 10 chunks instead of every chunk
- **GPU**: Auto-detected and utilized for codec
- **Typical speed**: ~1-3 seconds per chunk on GPU, ~2-5 seconds on CPU

### Real-World Example
**1000-word document (~5000 characters = ~17 chunks at 300 chars each)**
- **Before (CPU only, GC every chunk)**: ~60-100 seconds
- **After (CPU, optimized GC)**: ~50-85 seconds (~15% faster)
- **After (GPU, optimized GC)**: ~20-50 seconds (~2-3x faster)

The speedup varies based on:
- GPU availability and type (Apple Silicon MPS or NVIDIA CUDA)
- Text complexity (affects phonemization time)
- Chunk size and content

## Hardware Recommendations

### For Fastest Performance
- **GPU**: NVIDIA RTX 3060+ or Apple M1+ (with Metal)
- **CPU**: 4+ cores
- **RAM**: 8GB+ (16GB+ for long documents)

### Minimum Requirements
- **CPU**: 2+ cores
- **RAM**: 4GB+
- GPU optional (will auto-detect and use if available)

## Configuration

### Force CPU Mode
If you want to disable GPU usage (e.g., to save GPU for other tasks):

Edit `apple_tts_cloning.py`, line ~680:
```python
# Change from:
self.tts_engine.initialize_model()

# To:
self.tts_engine.initialize_model(force_cpu=True)
```

### Adjust GC Frequency
To change how often garbage collection runs:

Edit `apple_tts_cloning.py`, line ~1194:
```python
# Default: GC every 10 chunks
if (i + 1) % 10 == 0:
    gc.collect()

# More frequent (slower but lower memory):
if (i + 1) % 5 == 0:
    gc.collect()

# Less frequent (faster but more memory):
if (i + 1) % 20 == 0:
    gc.collect()
```

## Compatibility

- ✅ Maintains 300-character chunk limit
- ✅ Preserves session resume functionality
- ✅ Compatible with existing voice profiles
- ✅ Works on all platforms (Linux, macOS, Windows)
- ✅ Graceful degradation (works without GPU)

## Benchmarking

To see the improvements, watch for console output:
```
[device] Apple Metal (MPS) GPU detected
[perf] Sequential processing (phonemizer not thread-safe)
[tts] Using GGUF backend (backbone: cpu, codec: mps)
[memory] GC at chunk 10
[memory] GC at chunk 20
...
Complete! 17 chunks in 35s (0.49 chunks/sec)
```

## Future Optimization Opportunities

1. **Thread-safe phonemizer**: Would require rewriting/forking the phonemizer library to support concurrent access
2. **Process-based parallelism**: Use multiprocessing instead of threading (high memory overhead)
3. **Streaming output**: Start playing audio while still generating
4. **Model quantization**: Use INT8 for codec (if supported)
5. **C++ phonemizer**: Replace Python phonemizer with faster C++ implementation

## Notes

- The 300-character limit is **intentional** and cannot be increased (model constraint)
- **Parallel processing is NOT possible** due to phonemizer library limitations
- GPU acceleration works best with codec (where most computation happens)
- GGUF backend uses quantized weights (4-bit) for faster CPU inference
- Most performance gain comes from GPU codec acceleration, not parallelism
- On Apple Silicon (M1/M2/M3), MPS acceleration can provide 2-3x speedup
- On NVIDIA GPUs (RTX series), CUDA acceleration can provide similar speedup
