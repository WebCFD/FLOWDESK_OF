CPU_PARTITION_MAPPING = {
    1:  (1, 1, 1),
    2:  (2, 1, 1),
    3:  (3, 1, 1),
    4:  (2, 2, 1),
    6:  (3, 2, 1),
    8:  (2, 2, 2),
    9:  (3, 3, 1),
    12: (3, 2, 2),
    14: (7, 2, 1),
    16: (4, 2, 2),
    18: (3, 3, 2),
    20: (5, 2, 2),
    24: (4, 3, 2),
    27: (3, 3, 3),
    30: (5, 3, 2),
    32: (4, 4, 2),
    36: (3, 3, 4),
    40: (5, 4, 2),
    42: (7, 3, 2),
    45: (5, 3, 3),
    48: (4, 4, 3),
    50: (5, 5, 2),
    56: (7, 4, 2),
    60: (5, 4, 3),
    63: (7, 3, 3),
    64: (4, 4, 4),
}

def best_cpu_partition(n_cpus):
    # Filter keys less than or equal to n
    candidates = [k for k in CPU_PARTITION_MAPPING.keys() if k <= n_cpus]
    if not candidates:
        raise ValueError("No suitable partition found for input")
    # Find the max key <= n
    best_key = max(candidates)
    return best_key, CPU_PARTITION_MAPPING[best_key]