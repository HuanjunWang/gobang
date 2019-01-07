from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Activation, add
from keras.optimizers import Adam
from data import load_data


def make_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)

    x_i = x
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
    x = add([x, x_i])
    x = Activation(activation='relu')(x)

    x_i = x
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
    x = add([x, x_i])
    x = Activation(activation='relu')(x)

    x_i = x
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
    x = add([x, x_i])
    x = Activation(activation='relu')(x)

    x_i = x
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
    x = add([x, x_i])
    x = Activation(activation='relu')(x)

    x_i = x
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
    x = add([x, x_i])
    x = Activation(activation='relu')(x)

    x_i = x
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
    x = add([x, x_i])
    x = Activation(activation='relu')(x)

    x_i = x
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
    x = add([x, x_i])
    x = Activation(activation='relu')(x)

    x_i = x
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
    x = add([x, x_i])
    x = Activation(activation='relu')(x)

    x_i = x
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
    x = add([x, x_i])
    x = Activation(activation='relu')(x)


    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(units=32, activation='relu')(x)
    outputs = Dense(units=num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    adm = Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['acc'])
    return model


if __name__ == '__main__':
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = load_data(num_classes)
    input_shape = x_train.shape[1:]
    dnn = make_model(input_shape, num_classes)

    dnn.fit(x_train, y_train, epochs=10, batch_size=128)
    loss, acc = dnn.evaluate(x_test, y_test)
    print("Evaluation Loss:", loss, " Acc:", acc)
