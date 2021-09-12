from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator #lectura de imagenes
#10%-60%-30%
#book deep learning with python
def model_cnn():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPool2D(2,2))

    model.add(Conv2D(filters=62, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(2, 2))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(2, 2))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(2, 2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    optimizador = RMSprop(learning_rate=1e-4)
    loos_function = 'binary_crossentropy'
    model.compile(optimizer=optimizador, loss=loos_function, metrics=['acc'])
    model.summary()
    return model

model_cnn()

def pre_process_image():
    train_directorio = './data_set/training_dataset/'
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_directorio,
        target_size=(150, 150),
        color_mode='rgb',
        class_mode='binary',
        batch_size=5
    )
    validation_directorio = './data_set/training_dataset/'
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validataion_generator = validation_datagen.flow_from_directory(
        validation_directorio,
        target_size=(150, 150),
        color_mode='rgb',
        class_mode='binary',
        batch_size=5
    )

    return train_generator, validataion_generator

pre_process_image()

def train_model(model, train, validat):
    model.fit(train, steps_per_epoch=7, epochs= 25, validation_data=validat, validation_steps=32 )
    model.save_weights('model_entrenado.h5')

#nn_model = model_cnn()
#train_data, valid_data = pre_process_image()
#train_model(nn_model, train_data, valid_data)