import numpy as np

class MedianFilter:

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.half_window = window_size // 2
        self.org_vals = []

    def insert(self, value: float) -> None:
        
        if len(self.org_vals) > self.window_size:
            del self.org_vals[0] # remove first
        
        self.org_vals.append(value)

    def retrieve(self) -> float:
        return np.median(self.org_vals)