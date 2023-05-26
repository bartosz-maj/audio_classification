import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns  


from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score


from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import BaggingClassifier
from sklearn import svm 
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier



from google.colab import drive
import os, sys, re, pickle, glob
from tqdm import tqdm
import librosa
import librosa.display

drive.mount('/content/drive')

sample_path = '/content/drive/MyDrive/Data/all_audio/*.wav'
files = glob.glob(sample_path)

MLENDLS_df = pd.read_csv('./ml_gender.csv').set_index('file_id') 
MLENDLS_df.head()

def getPitch(x,fs,winLen=0.02):
  #winLen = 0.02 
  p = winLen*fs
  frame_length = int(2**int(p-1).bit_length())
  hop_length = frame_length//2
  f0, voiced_flag, voiced_probs = librosa.pyin(y=x, fmin=80, fmax=450, sr=fs,
                                                 frame_length=frame_length,hop_length=hop_length)
  return f0,voiced_flag 

def getXy(files,labels_file, scale_audio=False, onlySingleDigit=False):
  X,y =[],[]

  for file in tqdm(files):
    fileID = file.split('/')[-1]
    file_name = file.split('/')[-1]
    yi = labels_file.loc[fileID]['Gender']=='female'

    fs = None # if None, fs would be 22050
    x, sr = librosa.load(file,sr=fs)
    if scale_audio: x = x/np.max(np.abs(x))
    f0, voiced_flag = getPitch(x,sr,winLen=0.02)
      
    power = np.sum(x**2)/len(x)
    pitch_mean = np.nanmean(f0) if np.mean(np.isnan(f0))<1 else 0
    pitch_std  = np.nanstd(f0) if np.mean(np.isnan(f0))<1 else 0
    voiced_fr = np.mean(voiced_flag)

    xi = [pitch_mean,pitch_std]
    X.append(xi)
    y.append(yi)

  return np.array(X), np.array(y)

X,y= getXy(files, labels_file=MLENDLS_df, scale_audio=True, onlySingleDigit=True)

X_df = pd.DataFrame(X)
X_df = X_df.rename(columns = {0:"pitch_mean", 1:"pitch_sd"})
X_df.head()

X_y = pd.DataFrame(X)
X_y = X_y.rename(columns = {0:"pitch_mean", 1:"pitch_sd"})
X_y["labels"] = y
X_y.head()

X_train, X_val, y_train, y_val = train_test_split(X_df,y,test_size=0.3, random_state = 10)
