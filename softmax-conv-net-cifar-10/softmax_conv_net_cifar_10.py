import numpy as np
import tkinter as tk
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

# Network classification category titles
category = ['airplane','car','bird','cat','deer',
                'dog','frog','horse','ship','truck']
model = None

# Opens file browser to choose image to be classified,
# returns predicted category title
def predict():
    img_path = askopenfilename() # Opens file browser
    # Image is loaded as 32x32, converted to 32x32x3 array,
    # then expanded to fit models 4 dimension requirement,
    # pred = one-hot encoded prediction
    img = image.load_img(img_path, target_size=(32,32)) 
    test_image = image.img_to_array(img) 
    test_image = np.expand_dims(test_image, axis=0) 
    pred = model.predict(test_image, batch_size=1)
    cat_title = category[np.argmax(pred)]
    print('Image predicted to be ' + cat_title)
# Opens file browser to choose current model
def loadmodel():
    model_path = askopenfilename() # Opens file browser
    try:
        return load_model(model_path)
    except:
        print('Error loading model')

def savemodel():
    save_path = askopenfilename() # Opens file browser
    try:
        model.save(save_path)
    except:
        print('Error saving model')

# Creates a new model, TODO: separate and modularise
def newmodel():

    model = Sequential()

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3))) # 30, 30, 32
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

# Loads training data, compiles the model, runs 5x epochs of 
# 64 size minibatches of funn 50k training set (CIFAR-10)
def trainmodel():
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=0.0001, decay=1e-6),
                    metrics=['accuracy'])
    model.fit(X_train / 255.0, to_categorical(Y_train),
                batch_size=64,
                shuffle=True,
                epochs=5,
                validation_data=(X_test / 255.0, to_categorical(Y_test)),
                callbacks=[EarlyStopping(min_delta=0.001, patience=3)])
    model.summary()
    scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))
    print('Loss: %.3f' % scores[0])
    print('Accuracy: %.3f' % scores[1])

main_window = tk.Tk()
main_window.title('softmax-conf-net-cifar-10.py')

top_frame = tk.Frame(main_window).pack()
bottom_frame = tk.Frame(main_window).pack(side = "bottom")

load_btn = tk.Button(bottom_frame, text='Load CNN', command=loadmodel).pack(side='left')
train_btn = tk.Button(bottom_frame, text='Train', command=trainmodel).pack(side='left')
predict_btn = tk.Button(bottom_frame, text='Predict', command=predict).pack(side='left')
save_btn = tk.Button(bottom_frame, text='Save CNN', command=savemodel).pack(side='right')

model = loadmodel()
main_window.mainloop()
