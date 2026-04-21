import os


def get_unique_filename(path):
    """
    If path exists, append _v2, _v3, etc. until a non-existent path is found.
    Example: results/metrics.txt -> results/metrics_v2.txt
    """
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    counter = 2
    while True:
        new_path = f"{base}_v{counter}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1
