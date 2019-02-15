import numpy as np
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical

if __name__ == '__main__':

    running = True

    while running:

        action = input('Input: ').lower()

        if action.startswith('loadmodel'):

            model = load_model('softmax_conv_net_cifar_10_model_' + action.split(' ')[1] + '.h5')

        elif action.startswith('newmodel'):

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

        elif action.startswith('loaddata'):

            (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

        elif action.startswith('compile'):

            model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(lr=0.0001, decay=1e-6),
                          metrics=['accuracy'])

        elif action.startswith('train'):

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

        elif action.startswith('savemodel'):

            model.save('softmax_conv_net_cifar_10_model_' + action.split(' ')[1] + '.h5')