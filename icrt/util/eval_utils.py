import os
import pandas as pd
import numpy as np

class EvalLogger:
    def __init__(self, log_keys: list):
        self.log_keys = log_keys
        self._dict = {}
        for key in self.log_keys:
            self._dict.update({key: []})

    def add_kv(self, key, value):
        assert key in self.log_keys, f"Tried to log {key} but logger is initialized with keys {self.log_keys}"
        self._dict[key].append(value)

    def save(self, filename):
        df = pd.DataFrame(self._dict)
        # if the file exists, append to it
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            df.to_csv(filename, index=False)
        return

