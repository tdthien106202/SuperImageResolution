import os
import cv2
import sys
import glob
import time
import math
import argparse
import numpy as np
import tensorflow as tf 

from model import RFDNNet
from utils import *
from tensorflow.keras import Model, Input

import tkinter as tk
from tkinter import filedialog

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RFDNNet Parameters")

        # Test Path
        self.test_path_label = tk.Label(root, text="Test Path:")
        self.test_path_label.grid(row=0, column=0, padx=10, pady=10)
        self.test_path_entry = tk.Entry(root, width=40)
        self.test_path_entry.grid(row=0, column=1, padx=10, pady=10)
        self.browse_test_path_button = tk.Button(root, text="Browse", command=self.browse_test_path)
        self.browse_test_path_button.grid(row=0, column=2, padx=10, pady=10)

        # GPU
        self.gpu_label = tk.Label(root, text="GPU:")
        self.gpu_label.grid(row=1, column=0, padx=10, pady=10)
        self.gpu_entry = tk.Entry(root, width=40)
        self.gpu_entry.grid(row=1, column=1, padx=10, pady=10)

        # Weight Test Path
        self.weight_test_path_label = tk.Label(root, text="Weight Test Path:")
        self.weight_test_path_label.grid(row=2, column=0, padx=10, pady=10)
        self.weight_test_path_entry = tk.Entry(root, width=40)
        self.weight_test_path_entry.grid(row=2, column=1, padx=10, pady=10)
        self.browse_weight_test_path_button = tk.Button(root, text="Browse", command=self.browse_weight_test_path)
        self.browse_weight_test_path_button.grid(row=2, column=2, padx=10, pady=10)

        # RSAfilter
        self.RSAfilter_label = tk.Label(root, text="RSAfilter:")
        self.RSAfilter_label.grid(row=3, column=0, padx=10, pady=10)
        self.RSAfilter_entry = tk.Entry(root, width=40)
        self.RSAfilter_entry.grid(row=3, column=1, padx=10, pady=10)

        # Filter
        self.filter_label = tk.Label(root, text="Filter:")
        self.filter_label.grid(row=4, column=0, padx=10, pady=10)
        self.filter_entry = tk.Entry(root, width=40)
        self.filter_entry.grid(row=4, column=1, padx=10, pady=10)

        # Feat
        self.feat_label = tk.Label(root, text="Feat:")
        self.feat_label.grid(row=5, column=0, padx=10, pady=10)
        self.feat_entry = tk.Entry(root, width=40)
        self.feat_entry.grid(row=5, column=1, padx=10, pady=10)

        # Scale
        self.scale_label = tk.Label(root, text="Scale:")
        self.scale_label.grid(row=6, column=0, padx=10, pady=10)
        self.scale_entry = tk.Entry(root, width=40)
        self.scale_entry.grid(row=6, column=1, padx=10, pady=10)

        # Run Button
        self.run_button = tk.Button(root, text="Run", command=self.run_model)
        self.run_button.grid(row=10, column=1, pady=20)

    def browse_test_path(self):
        test_path = filedialog.askdirectory()
        self.test_path_entry.delete(0, tk.END)
        self.test_path_entry.insert(0, test_path)

    def browse_weight_test_path(self):
        weight_test_path = filedialog.askopenfilename(filetypes=[("H5 Files", "*.h5")])
        self.weight_test_path_entry.delete(0, tk.END)
        self.weight_test_path_entry.insert(0, weight_test_path)

    def run_model(self):
        # Get the values entered by the user
        test_path = self.test_path_entry.get()
        gpu = self.gpu_entry.get()
        weight_test_path = self.weight_test_path_entry.get()
        RSAfilter = int(self.RSAfilter_entry.get())
        filter_val = int(self.filter_entry.get())
        feat = int(self.feat_entry.get())
        scale = int(self.scale_entry.get())

        # Create a dictionary of parameters to pass to the main script
        config = {
            'test_path': test_path,
            'gpu': gpu,
            'weight_test_path': weight_test_path,
            'RSAfilter': RSAfilter,
            'filter': filter_val,
            'feat': feat,
            'scale': scale
        }

        # Create RFDNNet model
        rfanet_x = RFDNNet()
        x = Input(shape=(None, None, 3))
        out = rfanet_x.main_model(x, 3)
        rfa = Model(inputs=x, outputs=out)
        rfa.summary()

        # Load pre-trained weights
        rfa.load_weights(config['weight_test_path'])

        # Run the model
        run(config, rfa)

if __name__ == "__main__":
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()