import os
import cv2

from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visualize_trajectory(obs):
    # for d in obs:
    print("Starting to visualize...")
    if (isinstance(obs, list) or isinstance(obs, np.ndarray)):
        obs = {"rgb_static": obs}
    index = 0
    while True:
        img = None
        if isinstance(obs, dict):
            img = obs["rgb_static"][index][:, :, ::-1]
        else:
             raise ValueError(f"Unrecognized type of obs: {type(obs)}")
        cv2.imshow("image", img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
        elif key == 83:  # Right arrow
            index = (index + 1) % len(obs["rgb_static"])
        elif key == 81:  # Left arrow
            index = (len(obs["rgb_static"]) + index - 1) % len(obs["rgb_static"])
        else:
            print(f'Unrecognized keycode "{key}"')
        if index % len(obs["rgb_static"]) == 0:
            print("End of dataset reached")
            break
    cv2.destroyAllWindows()
    return


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

