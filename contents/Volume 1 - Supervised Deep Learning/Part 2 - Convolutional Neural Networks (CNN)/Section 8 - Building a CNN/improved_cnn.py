from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras import backend
import os

# Image dimensions
img_width, img_height = 150, 150


def create_model(p, input_shape=(32, 32, 3)):
    # Initialising the CNN
    model = Sequential()
    # Convolution + Pooling Layer
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening
    model.add(Flatten())
    # Fully connection
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p / 2))
    model.add(Dense(1, activation='sigmoid'))

    # Compiling the CNN
    optimizer = Adam(lr=1e-3)
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
    return model


"""
    Fitting the CNN to the images.
"""
def run_training(bs=32, epochs=10, workers=-1, max_q_size=10):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size=(img_width, img_height),
                                                     batch_size=bs,
                                                     class_mode='binary')

    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(img_width, img_height),
                                                batch_size=bs,
                                                class_mode='binary')

    model = create_model(p=0.6, input_shape=(img_width, img_height, 3))
    model.fit_generator(training_set,
                        steps_per_epoch=8000 / bs,
                        epochs=epochs,
                        validation_data=test_set,
                        validation_steps=2000 / bs,
                        workers=workers,
                        max_q_size=max_q_size)
    return model

def main():
    model = run_training(bs=32, epochs=100, workers=12, max_q_size=100)

    # saving model
    script_dir = os.path.dirname(__file__)
    model_backup_path = os.path.join(script_dir, '../dataset/cat_or_dogs_model.h5')
    model.save(model_backup_path)
    print("Model saved to", model_backup_path)

    backend.clear_session()

""" Main """
if __name__ == "__main__":
    main()