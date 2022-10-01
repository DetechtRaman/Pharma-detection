import os
from keras.regularizers import l2
from keras.layers import Conv1D, BatchNormalization,Dropout, Activation, Flatten, Dense, ZeroPadding1D, Input, MaxPooling1D, Add
from keras.models  import load_model, Model
import pickle
import numpy
import pandas as pd
from sklearn import manifold
import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform as glorot_uniform
from tensorflow.keras.utils import to_categorical
from scipy.signal import savgol_filter
import keras
from sklearn.model_selection import KFold
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
import re

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (logs.get('val_loss') < logs.get('loss') and epoch >10 and logs.get('val_loss') < 0.01):
            print("\nOptimized model achieved, so stopping training!!")
            self.model.stop_training = True

min_value=100
max_value=1000
start_wave, end_wave =650, 1700
colors = []
dir= "\\report\\"
datadir="pharma_data\\"
marker = ['v', '+', '1','>', 'o', '<', 'x', '*', 'v','o','1','+', 'x']
sg_winsize=[3, 5, 7, 9, 11]

def shift_left(X, shift=1):
    Y=np.copy(X)
    for ii in range(X.shape[1]-shift):
        Y[0,ii]=X[0, ii+shift]
        # print("shifted left")
    return Y

def shift_right(X, shift=1):
    Y=np.copy(X)
    for ii in range(X.shape[1]-shift):
        # print(ii)
        Y[0, ii+shift]=X[0, ii]
        # print("shifted right")
    return Y

def first_derivative(X,Y):
    dx=[]
    dy= []
    for i in range(X.shape[0]-1):
        dy.append(Y[i+1]-Y[i])
        dx.append((X[i+1]+X[i])/2)
    return dx, dy

def standardize(spectrum, wave, start_index, end_index, window_length=11):
    # Truncate ROI
    norm_spectrum = spectrum[start_index: end_index + 1]
    wave = np.asarray(wave[start_index: end_index + 1])

    # Remove high-frequency noise
    norm_spectrum = savgol_filter(x=norm_spectrum, window_length=window_length, polyorder=2, mode='nearest')

    # Remove background noise
    wave, norm_spectrum = first_derivative(wave, norm_spectrum)
    norm_spectrum= np.asarray(norm_spectrum)
    norm_spectrum = norm_spectrum *100

    spectrum = np.expand_dims(norm_spectrum, axis=0)
    return spectrum, wave

class DL_analysis:

    def create_augmented_data2(self):
        augY = []
        augX = []
        first = 0

        datadirpath= "dataset\\"+datadir
        for classdir in os.listdir(datadirpath):
            label = classdir
            print(label)

            dirpath = os.path.join(datadirpath, classdir)
            for file in os.listdir(dirpath):
                print("\t",file)
                filepath = os.path.join(dirpath, file)
                owave = np.loadtxt(filepath, usecols=0, dtype=int)
                owave = owave.tolist()
                ospectrum = np.loadtxt(filepath, usecols=1, dtype=float)
                start_index, end_index = owave.index(start_wave), owave.index(end_wave)

                for wsize in sg_winsize:

                    spectrum, wave = standardize(ospectrum, owave, start_index, end_index, window_length=wsize)

                    if first == 0:
                        augX = np.copy(spectrum)
                        first = 1
                    else:
                        augX = np.append(augX, spectrum, axis=0)
                    augY.append(label)

                    # Add random noise
                    exp = 1
                    for ee in range(1, 10):
                        noise = np.random.normal(0, 0.1 * exp, spectrum.shape[0])
                        ndata = spectrum + noise
                        
                        augX = np.append(augX, ndata, axis=0)
                        augY.append(label)
                        exp *= 0.1

                    #Shift data by 1 or 2 places
                    for sf in (range(1, 2)):
                        shifted_spectrum = shift_right(spectrum,sf)
                        augX = np.append(augX, shifted_spectrum, axis=0)
                        augY.append(label)

                        shifted_spectrum = shift_left(spectrum,sf)
                        augX = np.append(augX, shifted_spectrum, axis=0)
                        augY.append(label)

        print("augX.shape= ", augX.shape)
        print("augY.shape= ", len(augY))
        augX, augY = shuffle(augX, augY, random_state=42)
        print("Saving Augmented data")
        pickleout = open(dir+datadir+"count_gen_X.pickle", "wb")
        pickle.dump(augX, pickleout)
        pickleout.close()
        pickleout = open(dir+datadir+"count_gen_Y.pickle", "wb")
        pickle.dump(augY, pickleout)
        pickleout.close()

    def convolutional_block(self, X, f, filters, stage, block, s):

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        F1, F2, F3 = filters

        X_shortcut = X

        ##### MAIN PATH #####
        # First component of main path
        X = Conv1D(F1, 1, strides=s, name=conv_name_base + '2a', padding='valid',
                   kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(1e-4))(X)
        X = BatchNormalization(name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path
        X = Conv1D(F2, f, strides=1, name=conv_name_base + '2b', padding='same',
                   kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(1e-4))(X)
        X = BatchNormalization(name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path
        X = Conv1D(F3, 1, strides=1, name=conv_name_base + '2c', padding='valid',
                   kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(1e-4))(X)
        X = BatchNormalization(name=bn_name_base + '2c')(X)

        ##### SHORTCUT PATH ####
        X_shortcut = Conv1D(F3, 1, strides=s, name=conv_name_base + '1', padding='valid',
                            kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(1e-4))(X_shortcut)
        X_shortcut = BatchNormalization(name=bn_name_base + '1')(X_shortcut)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def ResNetv(self, input_shape, classes):

        X_input = Input(input_shape)

        X = ZeroPadding1D(3)(X_input)

        X = Conv1D(10, 3, strides=3, kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(1e-4),
                   name='conv1')(X)
        X = BatchNormalization(name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling1D(pool_size=3,strides=3, padding='same')(X)

        # Stage 2
        X = self.convolutional_block(X, f=3, filters=[10, 20, 20], stage=2, block='a', s=3)

        X = Dropout(0.01)(X)
        X = Flatten()(X)
        X = Dense(10*classes, activation="tanh", )(X)
        X = Dense(classes, activation=activation, name='fc' + str(classes))(X)

        model = Model(inputs=X_input, outputs=X, name='ResNet1')

        return model

    def createResNETmodel(self, i, t_x, val_x, t_y, val_y , num_class):

        Y = t_y
        X = numpy.array(t_x)

        X = X.reshape(X.shape[0], X.shape[1], -1)
        val_x = numpy.asarray(val_x)
        val_x = val_x.reshape(val_x.shape[0], val_x.shape[1], -1)

        print("After shaping X")
        print(X.shape)
        print("After shaping Y")
        print(Y.shape)
        num, size, dim = X.shape

        model = self.ResNetv(input_shape=(size, 1), classes= num_class)
        optimizer = tf.optimizers.Adam(lr=1e-4)
        model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

        callbacks = CustomCallback()
        model_checkpoint = ModelCheckpoint(dir+datadir+ 'Model_wine\\ResNetbest' + str(i) + '.hdf5',verbose=1,
                                           save_best_only=True)
        history = model.fit(X, Y, epochs=500, batch_size=40,  validation_split=0.2, verbose=2,
                            callbacks=[callbacks, model_checkpoint])

        score = model.evaluate(val_x, val_y)[1]
        print("Val Score : " + str(i) + "fold", score)
        model.save(dir+datadir+  "Model_wine\\ResNet_" + str(i) + "fold.hdf5")
        model.save_weights(dir+datadir+ "Model_wine\\Resnet_base_weights" + str(i) + ".hdf5")
        performance = pd.DataFrame(history.history['accuracy'])
        performance['Val_accuracy'] = history.history['val_accuracy']
        performance['Train_loss'] = history.history['loss']
        performance['val_loss'] = history.history['val_loss']
        performance.to_csv(dir+datadir+ "Model_wine\\ResNetperformance" + str(i) + ".csv", index=False)

        print("Saved model to disk")
        return score

    def cross_validation(self, n_folds):

        score_list = []
        wave, X, Y = self.load_aug_data()

        X, Y = shuffle(X, Y, random_state=0)

        Y = self.encode(Y).astype("int")

        kf = KFold(n_splits=n_folds, random_state=None, shuffle=False)

        i = 0
        num_class = len(np.unique(Y))
        global activation, loss

        if num_class > 2:
            Y = to_categorical(Y, num_class)

            activation = "softmax"
            loss = "categorical_crossentropy"
        else:
            activation = "sigmoid"
            loss = "binary_crossentropy"
            num_class=1

        print("activation= ", activation)
        for train_index, test_index in kf.split(X):
            print(i, "- fold cross validation")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            print("X_train.shape= ", X_train.shape)
            print("y_train.shape= ", y_train.shape)
            print("X_test.shape= ", X_test.shape)
            print("y_test.shape= ", y_test.shape)
            score = self.createResNETmodel(i, X_train, X_test, y_train, y_test, num_class)
            score_list.append(score)
            i += 1
            break
        print("Mean Accuracy= {:f}%".format(np.mean(score_list) * 100))

    def verify_feature_extraction(self, modelfname, X, Y):

        model = load_model(modelfname)
        head, tail = os.path.split(modelfname)
        modelname = os.path.splitext(tail)[0]
        X = numpy.asarray(X)
        X = X.reshape(X.shape[0], X.shape[1], -1)

        inp = model.inputs
        layernames = []

        for layer in model.layers:
            if ( "res" in layer.name):
                layernames.append(layer.name)
        last = len(layernames) - 1
        print("Creating t-SNE for last layer:", layernames[last])
        enc = Model(inp, model.get_layer(layernames[last]).output)
        print("Compiling")
        optimizer = tf.optimizers.Adam(lr=1e-4)
        enc.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['mae'])
        print("Predicting")
        P = enc.predict(X)

        tsne = manifold.TSNE(n_components=3, init='random')
        P = P.reshape(P.shape[0], P.shape[1] * P.shape[2])
        Ptsne = tsne.fit_transform(P)
        Ptsne = numpy.asarray(Ptsne).astype('float32')

        data = pd.DataFrame(Ptsne)
        data["Label"] = Y
        self.plot_tSNE(data, modelname, str(last))

    def plot_tSNE(self, data, name, lindex):
        Y = data["Label"]
        data = np.asarray(data)
        X = data[:, :-1]
        plt.rcParams["figure.figsize"] = [6, 6]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        classes= np.unique(Y)
        cmap = 'jet_r'
        colormap = cm.get_cmap(cmap, len(classes))
        colors = colormap(np.linspace(0, 1, len(classes)))
        for target, color in zip(classes, colors):
            i = Y == target
            class_i = classes.tolist().index(target)
            ax.scatter3D(X[i, 0], X[i, 1],X[i, 2], c=color, label=target, marker=marker[class_i] )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        plt.legend(classes)
        plt.savefig(dir+datadir+"Model_wine\\" + name + "t-SNE_plot_layer"+lindex+".png")
        plt.show()

    def validate_model(self, testdir, model):
        print("\n")
        print(testdir)

        picklein = open(dir + datadir + "count_gen_Y.pickle", "rb")
        Y = pickle.load(picklein)
        class_list = np.unique(Y)
        n_classes= len(class_list)

        spectra=[]
        true_y=[]
        for file in os.listdir(testdir):
            filepath = os.path.join(testdir, file)
            wave = np.loadtxt(filepath, usecols=0, dtype=int)
            owave = wave.tolist()
            ospectrum = np.loadtxt(filepath, usecols=1, dtype=float)

            start_index, end_index = owave.index(start_wave), owave.index(end_wave)

            filename = file.split(".txt")
            match = re.match(r"([a-z]+)([0-9]+)", filename, re.I)
            filename = match.groups()[0]
            label= filename

            spectrum, wave = np.asarray(ospectrum), np.asarray(owave)
            spectrum, wave = standardize(spectrum, wave, start_index, end_index, window_length=9)
            spectra.append(spectrum)
            true_y.append(label)

        pred_y = model.predict(spectra)

        self.plot_ROC_curve( pred_y,true_y, n_classes)

    def plot_ROC_curve(self, pred_y,true_y, n_classes):

        cmap = 'jet_r'
        colormap = cm.get_cmap(cmap, n_classes)
        colors = colormap(np.linspace(0, 1, n_classes))

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(true_y[:, i], pred_y[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        lw = 2

        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], color=colors[i],
                     lw=lw, label=n_classes[i] + '(AUC= %0.2f)' % roc_auc[i])

        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Performance Analysis by ROC-AUC')
        plt.legend(loc="lower right")
        plt.show()

    def encode(self, label):
        categories = np.unique(label)
        enc_label = np.zeros(len(label))
        print("categories= ", categories)
        print("categories.shape[0]= ", categories.shape[0])
        for i in range(categories.shape[0]):
            for j in range(len(label)):
                if categories[i] == label[j]:
                    enc_label[j] = i
        return enc_label

    def load_aug_data(self):
        picklein = open(dir+datadir+"count_gen_X.pickle", "rb")
        X = pickle.load(picklein)
        picklein = open(dir+datadir+"count_gen_Y.pickle", "rb")
        Y = pickle.load(picklein)
        X = numpy.asarray(X).astype("float32")
        data = np.asarray(pd.read_csv("sample.txt", header=None))
        wave = data[0, :-1].astype("int")
        return wave, X, Y

n_folds=5

obj= DL_analysis()
obj.create_augmented_data2()
obj.cross_validation(n_folds)

i=1
modelpath= dir+datadir+"Model_wine\\ResNetbest"+str(i)+".hdf5"
model = load_model(modelpath)
wave, X, Y = obj.load_aug_data()
X, Y = shuffle(X, Y, random_state=0)
X, Y = X[:900, :], Y[:900]
obj.verify_feature_extraction(modelpath, X, Y)

testdirlist = [ "test_data_jul25"]
for testdir in testdirlist:
   obj.validate_model("dataset\\" + testdir, model)