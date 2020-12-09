import tensorflow as tf
from tensorflow.keras import datasets,layers, models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout,Input, AveragePooling2D, Activation,Conv2D, MaxPooling2D, BatchNormalization,Concatenate
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


global train_images
global train_labels
global test_images
global test_labels

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print("Dataset loading Done")

class CNN():

    def __init__(self):
        pass
    
    def model_start(self):
        self.model_create()
        self.model_summary()
        var = self.model_compile()
        print("Model Creation Done")
        
    def model_create(self):

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Dropout(0.3))
        self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Dropout(0.4))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))

    def model_summary(self):
        self.model.summary()


    def model_compile(self):
        self.model.compile(optimizer='adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        return 0
        
    
    def train(self):
        global train_images
        global train_labels
        global test_images
        global test_labels
        self.history = self.model.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels))


    def model_performance_graph(self):
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')


    def model_evaluate(self):
        test_loss, test_acc = self.model.evaluate(test_images,  test_labels, verbose=2)

        print(test_acc)
        return test_acc


    def predict_new(self, image_path):
        img=mpimg.imread(image_path)
        ret_class = class_names[self.model.predict(np.expand_dims(img,axis=0)).argmax()]
        print(ret_class)
        return ret_class
    
    def model_save(self):
        self.model.save('model')
        
    def model_load(self):
        self.model = tf.keras.models.load_model('model')