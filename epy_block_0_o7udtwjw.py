import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    """
    Custom GNU Radio Python block:
    - Accepts vectors of 1024 complex64 samples
    - Extracts real and imag parts into 2048-length feature vector
    - Appends fixed label to each vector
    - Saves all to NPZ with 'features' and 'labels' arrays
    """

    def __init__(self, filename="output.npz", label=0):
        gr.sync_block.__init__(
            self,
            name="Labeled NPZ File Sink",
            in_sig=[(np.complex64, 1024)],
            out_sig=[]
        )

        self.filename = filename
        self.label = label
        self.features = []
        self.labels = []

    def work(self, input_items, output_items):
        in0 = input_items[0]  # Shape: (N, 1024)

        for vec in in0:
            real = np.real(vec)
            imag = np.imag(vec)
            feature_vector = np.concatenate((real, imag))  # Shape: (2048,)
            self.features.append(feature_vector)
            self.labels.append(self.label)

        return len(in0)

    def stop(self):
        # Save features and labels to NPZ
        if self.features:
            features_array = np.array(self.features, dtype=np.float32)  # Shape: (N, 2048)
            labels_array = np.array(self.labels, dtype=np.int32)        # Shape: (N,)
            np.savez(self.filename, features=features_array, labels=labels_array)
            print(f"Saved {len(labels_array)} vectors to {self.filename}")
        return True
