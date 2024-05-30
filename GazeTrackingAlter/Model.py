from keras import Sequential, regularizers, optimizers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.models import load_model


def NewModel():
    """
    :return: Model

    创建新的模型
    """
    Model = Sequential(name="Accelerator")

    Model.add(Input(shape=(25, 50, 3)))
    Model.add(Conv2D(20, 5, activation='relu'))
    Model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    Model.add(Flatten())
    Model.add(Dropout(0.2))
    Model.add(Dense(2, activation='tanh'))

    Adam = optimizers.Adam(learning_rate=0.005)
    Model.compile(optimizer=Adam, loss='mean_squared_error', metrics=['accuracy'])

    return Model

# Model.add(Input(shape=(25, 50, 3)))
# Model.add(Conv2D(32, (3, 3), activation='relu'))
# Model.add(MaxPooling2D((2, 2)),)
# Model.add(Conv2D(64, (3, 3), activation='relu'))
# Model.add(MaxPooling2D((2, 2)),)
# Model.add(Flatten())
# Model.add(Dense(128, activation='relu'))
# Model.add(Dense(64, activation='relu'))
# Model.add(Dense(2))
#
# Model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Model = Sequential(name="Accelerator")
#
# Model.add(Input(shape=(25, 50, 3)))
# Model.add(Conv2D(20, 5, activation='relu', kernel_regularizer=regularizers.l1_l2(0.001)))
# Model.add(Conv2D(40, 5, activation='relu', kernel_regularizer=regularizers.l1_l2(0.001)))
# Model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# Model.add(Flatten())
# Model.add(Dropout(0.3))
# Model.add(Dense(2, activation='tanh'))
#
# Adam = optimizers.Adam(learning_rate=0.005)
# Model.compile(optimizer=Adam, loss='mean_squared_error', metrics=['accuracy'])
