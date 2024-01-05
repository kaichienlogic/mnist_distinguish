import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
from tensorflow.python.keras.utils import np_utils
np.random.seed(10)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
(x_train_image, y_train_label), \
(x_test_image, y_test_label)=mnist.load_data()
print('train_data=',len(x_train_image))
print('test_data=',len(x_test_image))

x_Train=x_train_image.reshape(60000, 784).astype('float32')
x_Test=x_test_image.reshape(10000, 784).astype('float32')

x_Train_normalize=x_Train/255
x_Test_normalize=x_Test/255

y_Train_OneHot=np_utils.to_categorical(y_train_label)
y_Test_OneHot=np_utils.to_categorical(y_test_label)

model=Sequential()
model.add(Dense(units=256,
                input_dim=784,
                kernel_initializer='normal',
                activation='relu'))

model.add(Dense(units=10,
                kernel_initializer='normal',
                activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

train_history=model.fit(x=x_Train_normalize,
                        y=y_Train_OneHot, validation_split=0.2,
                        epochs=10, batch_size=200, verbose=2)

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
show_train_history(train_history, 'accuracy', 'val_accuracy')
show_train_history(train_history, 'loss', 'val_loss')
scores=model.evaluate(x_Test_normalize, y_Test_OneHot)
print()
print('accuracy=',scores[1])
#prediction=model.predict_classes(x_Test)
prediction=model.predict(x_Test)
prediction=np.argmax(prediction,axis=1)
prediction


def plot_images_labels_predict(images, labels, prediction, idx, num=10):  
    fig = plt.gcf()  
    fig.set_size_inches(12, 14)  
    if num > 25: num = 25  
    for i in range(0, num):  
        ax=plt.subplot(5,5, 1+i)  
        ax.imshow(images[idx], cmap='binary')  
        title = "l=" + str(labels[idx])  
        if len(prediction) > 0:  
            title = "l={},p={}".format(str(labels[idx]), str(prediction[idx]))  
        else:  
            title = "l={}".format(str(labels[idx]))  
        ax.set_title(title, fontsize=10)  
        ax.set_xticks([]); ax.set_yticks([])  
        idx+=1  
    plt.show()  
plot_images_labels_predict(x_test_image, y_test_label, prediction, idx=240)
