import psutil

def get_memory_usage_mb():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)
