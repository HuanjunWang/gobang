from resnet_lib import *
from data import *

from keras.optimizers import Adam

num_classes = 10
(x_train, y_train), (x_test, y_test) = load_data(num_classes)
input_shape = x_train.shape[1:]

print(input_shape)

resnet = ResnetBuilder.build_resnet_18((input_shape[2], input_shape[0], input_shape[1]), 10)

print(resnet.summary())

adm = Adam(lr=1e-3)
resnet.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['acc'])

resnet.fit(x_train, y_train, epochs=1, batch_size=128)
loss, acc = resnet.evaluate(x_test, y_test)
print("Evaluation Loss:", loss, " Acc:", acc)

