from . import gaussian_background

try:
    import tinycudann

    from . import gaussian_mvdream_background
except ImportError:
    print("tinycudann is not installed or could not be imported.")
