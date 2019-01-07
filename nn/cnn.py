import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import Adam


def load_data(num_classes):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


def make_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.summary()

    adm = Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['acc'])

    return model


if __name__ == '__main__':
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = load_data(num_classes)
    input_shape = x_train.shape[1:]
    cnn = make_model(input_shape, num_classes)

    cnn.fit(x_train, y_train, epochs=1, batch_size=128)
    loss, acc = cnn.evaluate(x_test, y_test)
    print("Evaluation Loss:", loss, " Acc:", acc)
