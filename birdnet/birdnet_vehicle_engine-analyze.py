# general packages
import warnings
warnings.filterwarnings(action='ignore')
import os

# basic packages
import numpy as np              # adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import pandas as pd             # library providing high-performance, easy-to-use data structures and data analysis tools (Dataframe)
import matplotlib.pyplot as plt # a comprehensive library for creating static, animated, and interactive visualizations
plt.rcParams.update({'figure.max_open_warning': 0})
plt.style.use("default")


import subprocess

DATA_PATH = r'C:\Users\phano\Documents\_CDD LECA\Ecriture\Code\OPH-STAM2-LECA\birdnet'
SUPP_AUDIO_PATH = r'C:\Users\phano\Documents\_MoBI 2024\LECA2025\Audio\Audio-3s-32000'
SAVE_PATH = r'C:\Users\phano\Documents\_CDD LECA\Ecriture\Code\OPH-STAM2-LECA\birdnet\save'
PACKAGE_PATH = r'C:\Users\phano\Documents\_CDD LECA\Pipeline'
MODEL_PATH = r'C:\Users\phano\Documents\_CDD LECA\Ecriture\Code\OPH-STAM2-LECA\birdnet\model'

os.chdir(PACKAGE_PATH)

cmd_analyze = [
  "python", "-m", "BirdNET-Analyzer.birdnet_analyzer.analyze", 
  f"{SUPP_AUDIO_PATH}", 
  "-c",f"{MODEL_PATH}/custom_birdnet_vehicle_engine_b.tflite",
  "-o", f"{SAVE_PATH}/audio3s_b2", 
  "--min_conf", "0.1", 
  "--rtype", "csv"]

result = subprocess.run(cmd_analyze, capture_output=True, text=True)
print("RETURN CODE:", result.returncode)
print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)

# subprocess.run(cmd_analyze, capture_output=True, text=True)