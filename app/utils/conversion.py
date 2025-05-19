# utils/conversion.py
import numpy as np

def sanitize_numpy_types(data: dict) -> dict:
    for k, v in data.items():
        if isinstance(v, (np.float64, np.float32)):
            data[k] = float(v)
        elif isinstance(v, (np.int64, np.int32)):
            data[k] = int(v)
    return data