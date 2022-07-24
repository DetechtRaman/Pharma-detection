
import PySimpleGUI as sg
import pickle
import numpy as np
from keras.models  import  load_model
from scipy.signal import savgol_filter
import os
sg.theme('BluePurple')
# sg.theme('Default')

layout = [
    # [sg.Text('Your typed characters appear here:')],
           # sg.Text(size=(15, 1), key='-OUTPUT-')],
          # [sg.Input(key='-IN-')],
        [sg.Input(sg.user_settings_get_entry('-filename-', ''),enable_events=True , key='-IN-'), sg.FileBrowse('FileBrowse')],
          # [sg.Button('Predict',disabled=True)],
          [sg.Button('Verify')],
        [sg.Text(size=(30, 1), key='-OUTPUT-',  font='Arial 14', text_color='blue')]
        ]

# window = sg.Window('Alcohol Counterfeit Detection', layout, size=(400,800))
window = sg.Window('Pharmaceutical Drug Detection Prototype 3',  layout)
model = load_model("DL_model3.hdf5")

def first_derivative(X,Y):
    dx=[]
    dy= []
    for i in range(X.shape[0]-1):
        dy.append(Y[i+1]-Y[i])
        dx.append((X[i+1]+X[i])/2)
    return dx, dy

def preprocess(spectrum, wave, window_length=9):

    norm_spectrum = savgol_filter(x=spectrum, window_length=window_length, polyorder=2, mode='nearest')
    norm_spectrum = np.asarray(norm_spectrum)
    wave = np.asarray(wave)
    wave, norm_spectrum= first_derivative(wave, norm_spectrum)
    norm_spectrum = np.asarray(norm_spectrum)
    wave = np.asarray(wave)
    norm_spectrum=norm_spectrum*10
    spectrum = np.expand_dims(norm_spectrum, axis=0)
    return spectrum, wave

while True:
    event, values = window.read()
    print(event)
    filename = values['FileBrowse']
    start_wave, end_wave = 330, 1800
    # model = pickle.load(open('D:\\IISC\\Vishnu\\win_project\\report_01_20\\RFC_glassbottle_org_count.sav', 'rb'))

    # if event == sg.WIN_CLOSED or event == "Exit":
    if event == sg.WIN_CLOSED:
        break

    elif event == '-IN-':

        window['-OUTPUT-'].update("")

        if not os.path.exists(filename):
            window['-OUTPUT-'](text_color='black')
            window['-OUTPUT-'].update("This file does not exist.")
    #  [sg.Button('basic AI'), sg.Button('advanced AI')],
    elif event == 'Verify':

        # window['-OUTPUT-'].update(" ")
        if filename == "":
            window['-OUTPUT-'](text_color='black')
            window['-OUTPUT-'].update("Select the file.")

        else:
            print("Opening spectrum from ", filename)
            wave = np.loadtxt(filename, usecols=0, dtype=int)
            wave = wave.tolist()
            spectrum = np.asarray(np.loadtxt(filename, usecols=1, dtype=float))
            print("Preprocessing spectrum... ")
            start_index, end_index = wave.index(start_wave), wave.index(end_wave)
            wave = wave[start_index: end_index+1]
            spectrum = spectrum[start_index: end_index+1]
            print(spectrum.shape)
            spectrum= preprocess(spectrum, wave)
            spectrum= np.asarray(spectrum)[0]
            # spectrum = np.expand_dims(spectrum, axis=0)
            # spectrum= np.expand_dims(spectrum, axis=1)
            print(spectrum.shape)
            print("Predicting using DL model...")
            result = model.predict(spectrum)[0]
            result = np.round(result, 2)

            print("result= ",result)
            if result <=0.5:
                label, color = "a counterfeit!", "maroon"
            else:
                label, color = "a genuine.", "darkblue"

            label = "The sample is " + label
            # window['-OUTPUT-'].Widget.configure(highlightcolor=color)
            window['-OUTPUT-'](text_color=color)
            window['-OUTPUT-'].update(label)

        # print("Its ", label)

# sg.theme_previewer()
window.close()
