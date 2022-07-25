from keras.models  import load_model
import PySimpleGUI as sg
import numpy as np

from sklearn.preprocessing import minmax_scale
from scipy.signal import savgol_filter
import os
sg.theme('BluePurple')


layout = [

        [sg.Input(sg.user_settings_get_entry('-filename-', ''),enable_events=True , key='-IN-'), sg.FileBrowse('FileBrowse')],

          [ sg.Button('Identify')],
        [sg.Text(size=(30, 1), key='-OUTPUT-',  font='Arial 14', text_color='blue')]
        ]

window = sg.Window('Pharmaceutical Drug Detection Prototype',  layout)
model1= load_model("class3_DLmodel.hdf5")

min_value=0
max_value=1000

def first_derivative(X,Y):
    dx=[]
    dy= []
    print("X.shape= ",X.shape)
    print("Y.shape= ",Y.shape)
    for i in range(X.shape[0]-1):
        dy.append(Y[i+1]-Y[i])
        dx.append((X[i+1]+X[i])/2)
    return dx, dy

while True:
    event, values = window.read()
    print(event)
    filename = values['FileBrowse']
    start_wave, end_wave =650, 1700
    # model = pickle.load(open('D:\\IISC\\Vishnu\\win_project\\report_01_20\\RFC_glassbottle_org_count.sav', 'rb'))

    # if event == sg.WIN_CLOSED or event == "Exit":
    if event == sg.WIN_CLOSED:
        break

    elif event == '-IN-':

        window['-OUTPUT-'].update("")

        if not os.path.exists(filename):
            window['-OUTPUT-'](text_color='black')
            window['-OUTPUT-'].update("This file does not exist.")

    elif event == 'Identify':

        # window['-OUTPUT-'].update(" ")
        if filename == "":
            window['-OUTPUT-'](text_color='black')
            window['-OUTPUT-'].update("Select the file.")

        else:

            wave = np.loadtxt(filename, usecols=0, dtype=int)
            wave = wave.tolist()
            spectrum = np.loadtxt(filename, usecols=1, dtype=float)

            start_index, end_index = wave.index(start_wave), wave.index(end_wave)

            spectrum = spectrum[start_index: end_index+1]
            wave = np.asarray(wave[start_index: end_index + 1])
            spectrum = savgol_filter(x=spectrum, window_length=5, polyorder=2, mode='nearest')

            wave, spectrum = first_derivative(wave, spectrum)
            spectrum = np.asarray(spectrum)
            spectrum = spectrum * 100
            spectrum = np.expand_dims(spectrum, axis=0)

            color = "darkblue"
            result = model1.predict(spectrum)[0]
            result = np.round(result, 2)
            result = result * 100
            max = np.max(result)
            result = result.tolist()

            r_index = result.index(max)

            if r_index  ==0:
                label= "Calpol 500"
            elif r_index  == 1:
                label= "Dolo 650"
            else:
                label="Meftal Forte"
            label = "The sample is " + label

            window['-OUTPUT-'](text_color=color)
            window['-OUTPUT-'].update(label)


window.close()
