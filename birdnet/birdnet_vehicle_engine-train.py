import os

DATA_PATH = r'C:\Users\phano\Documents\_CDD LECA\Ecriture\Code\OPH-STAM2-LECA\birdnet'
SUPP_AUDIO_PATH = r'C:\Users\phano\Documents\_CDD LECA\Ecriture\Code\OPH-STAM2-LECA\birdnet\data'
SAVE_PATH = r'C:\Users\phano\Documents\_CDD LECA\Ecriture\Code\OPH-STAM2-LECA\birdnet\save'
PACKAGE_PATH = r'C:\Users\phano\Documents\_CDD LECA\Pipeline'
MODEL_PATH = r'C:\Users\phano\Documents\_CDD LECA\Ecriture\Code\OPH-STAM2-LECA\birdnet\model'

# general packages
import warnings
warnings.filterwarnings(action='ignore')
import os

import matplotlib.pyplot as plt # a comprehensive library for creating static, animated, and interactive visualizations
plt.rcParams.update({'figure.max_open_warning': 0})
plt.style.use("default")



import subprocess

###### BIRDNET #######
# go to the directory
os.chdir(PACKAGE_PATH)


cmd_train = [
    "python", "-m", "BirdNET-Analyzer.birdnet_analyzer.train",
    f"{SUPP_AUDIO_PATH}/VEHICLE_ENGINEB_DATASET/TRAIN",
    # "--test_data",f"{SUPP_AUDIO_PATH}/VEHICLE_ENGINEB_DATASET/TEST",
    "--output", f"{MODEL_PATH}/custom_birdnet_vehicle_engine_b",
    "--mixup",
    "--hidden_units", "16",
    "--dropout", "0.33",
    "--crop_mode", "segments",
    "--overlap", "2",
]

result = subprocess.run(cmd_train, capture_output=True, text=True)
print("RETURN CODE:", result.returncode)
print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)


