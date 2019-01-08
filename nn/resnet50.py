from keras.optimizers import Adam, SGD
from data import load_data
from keras.applications.resnet50 import ResNet50

from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model

if __name__ == '__main__':
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = load_data(num_classes)
    input_shape = x_train.shape[1:]
    print(input_shape)

    basic_resnet = ResNet50(weights=None, include_top=False, input_shape=input_shape)

    x = basic_resnet.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=basic_resnet.input, outputs=predictions)

    for layer in basic_resnet.layers:
        layer.trainable = False


    adm = Adam(lr=1e-3)
    sgd = SGD(lr=1e-3, momentum=.8)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])

    model.fit(x_train, y_train, epochs=1, batch_size=128)
    loss, acc = model.evaluate(x_test, y_test)
    print("Evaluation Loss:", loss, " Acc:", acc)

# Basic network with imgnet weight
# Densy 1024 units
# lr 1e-4
# epochs 10
# training acc .71
# test acc .1
#

