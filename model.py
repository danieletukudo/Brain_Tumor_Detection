from typing import Tuple, Any
import os
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras import Model
import matplotlib.pyplot as plt


class myModel:

    def __init__(self) -> None:
        self.base_model = None
        self.input_shape = None

    def VGG16_model(self, input_shape:tuple) -> tf.keras.Model:

        self.base_model = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')

        for layer in self.base_model.layers:
            layer.trainable = False

        x = self.base_model.output
        flat = Flatten()(x)

        class_1 = Dense(4608, activation='relu')(flat)
        drop_out = Dropout(0.2)(class_1)
        class_2 = Dense(1152, activation='relu')(drop_out)
        output = Dense(2, activation='softmax')(class_2)

        model = Model(self.base_model.input, output)
        model.summary()

        # sgd = SGD(learning_rate=0.0001,  momentum = 0.9, nesterov = True)
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

        return model


class training:

    def __init__(self) -> None:
        self.train_generator = None
        self.val_generator = None

    def data_augmentation(self, train_data_path: os.path, val_data_path: os.path , input_shape:tuple) -> tuple[Any, Any]:
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           horizontal_flip=0.4,
                                           vertical_flip=0.4,
                                           rotation_range=40,
                                           shear_range=0.2,
                                           width_shift_range=0.4,
                                           height_shift_range=0.4,
                                           fill_mode='nearest')

        valid_data_gen = ImageDataGenerator(rescale=1.0 / 255)

        self.train_generator = train_datagen.flow_from_directory(train_data_path,
                                                                 batch_size=32,
                                                                 target_size= (input_shape[0],input_shape[1]),
                                                                 class_mode='categorical', shuffle=True,
                                                                 seed=42, color_mode='rgb')

        self.val_generator = valid_data_gen.flow_from_directory(val_data_path, batch_size=32,
                                                                target_size=(input_shape[0],input_shape[1]),
                                                                class_mode='categorical', shuffle=True, seed=42,
                                                                color_mode='rgb')

        return self.train_generator, self.val_generator

    def training(self,  epochs:int, train_path: os.path, val_path: os.path, input_shape:tuple,
                 trained_model_name:str ) -> None:


        data_generator = self.data_augmentation(train_path, val_path,input_shape)

        Model = myModel()
        model = Model.VGG16_model(input_shape)
        train_generator = data_generator[0]
        valid_generator = data_generator[1]


        history = model.fit(train_generator,
                            steps_per_epoch=10,
                            epochs = epochs,
                            validation_data=valid_generator)
        model.save(trained_model_name)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        fig.suptitle("Model Training ", fontsize=12)
        max_epoch = len(history.history['accuracy']) + 1
        epochs_list = list(range(1, max_epoch))

        ax1.plot(epochs_list, history.history['accuracy'], color='b', linestyle='-', label='Training Data')
        ax1.plot(epochs_list, history.history['val_accuracy'], color='r', linestyle='-', label='Validation Data')
        ax1.set_title('Training Accuracy', fontsize=12)
        ax1.set_xlabel('Epochs', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(frameon=False, loc='lower center', ncol=2)

        ax2.plot(epochs_list, history.history['loss'], color='b', linestyle='-', label='Training Data')
        ax2.plot(epochs_list, history.history['val_loss'], color='r', linestyle='-', label='Validation Data')
        ax2.set_title('Training Loss', fontsize=12)
        ax2.set_xlabel('Epochs', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(frameon=False, loc='upper center', ncol=2)
        plt.savefig("model_training.jpg", format='jpeg', dpi=100, bbox_inches='tight')
        plt.show()

    def quantize_model(self,model_name:str) -> None:

        model = tf.keras.models.load_model(model_name)
        tf.saved_model.save(model, 'saved_model_directory')
        saved_model_dir = "saved_model_directory"
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quant_model = converter.convert()
        with open('100quantized_model.tflite', 'wb') as f:
            f.write(tflite_quant_model)

        print("Model Quantization Done ")


if __name__ == "__main__":
    trained_model_name = "1model.h5"
    train = training()
    training = False

    if training:
        train.training(epochs=100, train_path="dataset/train",
                       val_path="dataset/val",
                       input_shape=(100, 100, 3),
                       trained_model_name="trained_model_name")

    train.quantize_model(trained_model_name)



