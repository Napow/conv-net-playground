import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing import image
from keras.utils import to_categorical

# Opens file browser to choose image to be classified,
# returns predicted category title
def predict():
    # Network classification category titles
    category = ['airplane','car','bird','cat','deer',
                'dog','frog','horse','ship','truck']
    img_path = askopenfilename() # Opens file browser
    # Image is loaded as 32x32, converted to 32x32x3 array,
    # then expanded to fit models 4 dimension requirement,
    # one-hot prediction converted to category title
    img = image.load_img(img_path, target_size=(32,32)) 
    test_image = image.img_to_array(img) 
    test_image = np.expand_dims(test_image, axis=0) 
    pred = model.predict(test_image, batch_size=1)
    cat_title = category[np.argmax(pred)]
    print('Image predicted to be ' + cat_title)

# Opens file browser, returns selected model
def loadmodel():
    model_path = askopenfilename() # Opens file browser
    try:
        return load_model(model_path)
    except:
        print('Error loading model')

# Opens file browser, saves current model as specified
def savemodel():
    save_path = askopenfilename() # Opens file browser
    try:
        model.save(save_path)
    except:
        print('Error saving model')

# Creates a new model BATCH NORM->CONV->CONV->DROPOUT->MAX POOLING->
# CONV->MAX POOLING->DROPOUT->CONV->MAX POOLING->FULLY CON->SOFTMAX 10
def newmodel():

    model = Sequential()

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 
                     input_shape=(32, 32, 3))) # 30, 30, 32
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) # 28, 28, 64
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2))) # 14, 14, 64

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu')) # 12, 12, 128
    model.add(MaxPooling2D(pool_size=(2, 2))) # 6, 6, 128
    model.add(Dropout(0.2))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu')) # 4, 4, 128
    model.add(MaxPooling2D(pool_size=(2, 2))) # 2, 2, 128

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
# Adds CONV2D layer to passed model, num_filters = number of filters in layer,
# 3x3 kernel size, 'same' padding, where multiple identical layers needed in
# succession num_layers can be passed to avoid needing to call several times
# in succession. Layers activated with ReLU.
# max_pooling can be passed as true to perform 2x2 max pooling following all
# layers requested being added to model.
def add_vgg_layer(model, num_filters, num_layers=1, max_pooling=False):
    for i in range(num_layers):
        model.add(Conv2D(num_filters, kernel_size=(3, 3), padding='same',
                         activation='relu'))
    if max_pooling:
        model.add(MaxPooling2D(pool_size=(2, 2)))
    return model

# Creates standard VGG-16 model, 224x224x3 input, 1000 node softmax output
# As derived from https://arxiv.org/abs/1409.1556
def newmodel_vgg():

    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu',
                     input_shape=(224, 224, 3))) # 224, 224, 64
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same',
                     activation='relu')) # 224, 224, 64
    
    model.add(MaxPooling2D(pool_size=(2, 2))) # 112, 112, 64
    
    #model = add_vgg_layer(model, 128, num_layers = 2, max_pooling = True)

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same',
                     activation='relu')) # 112, 112, 128
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same',
                     activation='relu')) # 112, 112, 128
    
    model.add(MaxPooling2D(pool_size=(2, 2))) # 56, 56, 128
    
    #model = add_vgg_layer(model, 256, num_layers = 3, max_pooling = True)
    
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same',
                     activation='relu')) # 56, 56, 256
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same',
                     activation='relu')) # 56, 56, 256
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same',
                     activation='relu')) # 56, 56, 256
    
    model.add(MaxPooling2D(pool_size=(2, 2))) # 28, 28, 256
    
    #model = add_vgg_layer(model, 512, num_layers = 3, max_pooling = True)
    
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                     activation='relu')) # 28, 28, 512
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                     activation='relu')) # 28, 28, 512
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                     activation='relu')) # 28, 28, 512
    
    model.add(MaxPooling2D(pool_size=(2, 2))) # 14, 14, 512
    
    #model = add_vgg_layer(model, 512, num_layers = 3, max_pooling = True)
    
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                     activation='relu')) # 14, 14, 512
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                     activation='relu')) # 14, 14, 512
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                     activation='relu')) # 14, 14, 512
    
    model.add(MaxPooling2D(pool_size=(2, 2))) # 7, 7, 512    

    # Flatten 7x7x512 output to 1x1x4096, pass through 2x 4096 node fully
    # connected layers then 1000 class softmax classifier
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))    
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1000, activation='softmax'))

# Loads training data, compiles the model, runs training
# TODO: Browse for training data, add editor in GUI for other hyperparameters
def trainmodel(batch_size=64, epochs=1):
    # Load training data
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    # Configures model for training
    model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=0.0001, decay=1e-6),
                    metrics=['accuracy'])
    # Performs training on entered dataset
    fit_report = model.fit(X_train / 255.0, to_categorical(Y_train),
                batch_size=batch_size,
                shuffle=True,
                epochs=epochs,
                validation_data=(X_test / 255.0, to_categorical(Y_test)),
                callbacks=[EarlyStopping(min_delta=0.001, patience=3)])
    # Plot training accuracy over time
    plt.plot(fit_report.history['acc'])
    plt.plot(fit_report.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()
    
def testmodel():
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))
    print('Loss: %.3f' % scores[0])
    print('Accuracy: %.3f' % scores[1])
    
# Simple GUI for ease of navigation, frames to be separated
# into input and display sections
main_window = tk.Tk()
main_window.title('softmax-conf-net-cifar-10.py')

top_frame = tk.Frame(main_window).pack()
bottom_frame = tk.Frame(main_window).pack(side = "bottom")

load_btn = tk.Button(bottom_frame, text='Load CNN', 
                     command=loadmodel).pack(side='left')
train_btn = tk.Button(bottom_frame, text='Train', 
                      command=trainmodel).pack(side='left')
test_btn = tk.Button(bottom_frame, text='Test', 
                        command=testmodel).pack(side='left')
predict_btn = tk.Button(bottom_frame, text='Predict', 
                        command=predict).pack(side='left')
save_btn = tk.Button(bottom_frame, text='Save CNN', 
                     command=savemodel).pack(side='right')

# Request user for model to load on start
model = loadmodel()
main_window.mainloop()
