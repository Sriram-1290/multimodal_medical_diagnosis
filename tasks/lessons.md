# Lessons Learned

## Windows Environment Issues

### OMP: Error #15 (Duplicate OpenMP runtime)
- **Problem**: Encountered `OMP: Error #15` when running PyTorch training scripts on Windows. This is caused by multiple OpenMP libraries being initialized (e.g., from `torch` and `mkl`-backed `numpy`).
- **Solution**: Set `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` at the top of the entry point script (before heavy imports).
- **Pattern**: When setting up training pipelines for Windows environments, prefer adding this environment variable check early to avoid crashes.

## Dependency Management

### Missing Transitive Dependencies
- **Problem**: `ModuleNotFoundError: No module named 'tensorboard'` occurred even though it was expected to be available via `torch.utils.tensorboard`.
- **Solution**: Explicitly install `tensorboard` if it's not included in the base environment.
