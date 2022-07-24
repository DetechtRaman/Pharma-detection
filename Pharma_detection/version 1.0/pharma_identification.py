
import pickle
import numpy as np
from airPLS import airPLS
from sklearn.preprocessing import minmax_scale
from scipy.signal import savgol_filter
import sys


model1 = pickle.load(open('RFC_powdermodel.sav', 'rb'))
start_wave, end_wave = 330, 1800

def preprocess(x):
    baseline = airPLS(x)
    x = x - baseline
    x = savgol_filter(x, 9, 3, mode='nearest')
    x = minmax_scale(x, feature_range=(0, 1))
    return x

def main(argv):
    print("Analyzing...")
    filename= argv[0]
    wave = np.loadtxt(filename, usecols=0, dtype=int)
    wave = wave.tolist()
    spectrum = np.loadtxt(filename, usecols=1, dtype=float)

    start_index, end_index = wave.index(start_wave), wave.index(end_wave)
    spectrum = spectrum[start_index: end_index + 1]
    spectrum = preprocess(spectrum)
    spectrum = np.expand_dims(spectrum, axis=0)
    result = model1.predict(spectrum)
    if result == 0:
        label = "a counterfeit!"
    else:
        label= "a genuine."

    label = "The sample is " + label
    print(label)

if __name__ == "__main__":
   main(sys.argv[1:])