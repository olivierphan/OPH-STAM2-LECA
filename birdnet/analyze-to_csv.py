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
import pandas as pd
from glob import glob

import matplotlib.pyplot as plt # a comprehensive library for creating static, animated, and interactive visualizations
plt.rcParams.update({'figure.max_open_warning': 0})
plt.style.use("default")

import subprocess

#@title  { run: "auto", form-width: "1000px" }
# Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1.
min_confidence = 0.1    

# to manipulate date and time
from datetime import datetime
from pathlib import Path

# grab all birdnet output files
filelist = glob(SAVE_PATH+'/audio3s_c/'+'/*.csv', recursive = True)


filelist = [x for x in filelist if 'BirdNET_analysis_params.csv' not in x]



# create a dataframe with all anotations files in the directory
# add new columns similar to the output of BirdNET in order to compate both
# results
df_raw_birdnet = pd.DataFrame()

# list of columns
cols = ['filename', 'Start (s)','End (s)','Confidence', 'label' ]

for file in filelist:
  # read the csv file associated with the audio file
  df_rois = pd.read_csv(file, sep=',' )

  # if there is a detection, add a column with the filename and a column with the label
  if len(df_rois) > 0 :
    df_rois['filename'] = Path(file).parts[-1][:-20] + '.wav'
    if df_rois['Confidence'].max() > min_confidence :
      df_rois['label'] = [((x.split(" ", 1)[0][0:3]).lower()+(x.split(" ", 1)[1][0:3]).lower()) for x in df_rois['Scientific name']]
    else :
      df_rois['label'] = 'none'

  # else add a column with the filename and set the other columns to NaN
  else :
    df_rois = pd.DataFrame(columns=cols, dtype=float)
    df_rois.loc[0, 'filename'] = Path(file).parts[-1][:-20]+ '.wav'
    df_rois.loc[0, 'Start (s)'] = 0
    df_rois.loc[0, 'End (s)']   = 60
    df_rois.loc[0, 'label']     = 'none'
    df_rois.loc[0, 'Confidence']= 1

  # reorder the columns
  df_rois = df_rois[['filename',
                      'Start (s)',
                      'End (s)',
                      'Confidence',
                      'label'
                      ]]

  # add the annotations of the current file into the big dataframe with all
  # annotations
  df_raw_birdnet = pd.concat([df_raw_birdnet, df_rois], axis=0, ignore_index=True)

# display the first rows of the dataframe
df_raw_birdnet[df_raw_birdnet["label"]!="none"]["filename"].unique()

veheng_files = df_raw_birdnet
veheng_files.to_csv(os.path.join(DATA_PATH,'vehicle_engine_b-audio3s.csv'))

