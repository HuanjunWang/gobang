from keras.optimizers import Adam
from data import load_data
from keras.applications.resnet50 import ResNet50


if __name__ == '__main__':
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = load_data(num_classes)
    input_shape = x_train.shape[1:]
    resnet = ResNet50(weights=None, input_shape=input_shape, classes=num_classes, pooling='max')
    adm = Adam(lr=1e-4)
    resnet.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['acc'])

    resnet.fit(x_train, y_train, epochs=10, batch_size=128)
    loss, acc = resnet.evaluate(x_test, y_test)
    print("Evaluation Loss:", loss, " Acc:", acc)

# no pooling
# lr = 1e-4
# epochs = 20
# train acc = .91
# test acc = .45


