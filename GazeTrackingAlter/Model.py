from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

Model = Sequential(name="Accelerator")

Model.add(Input(shape=(30, 100, 3)))
Model.add(Conv2D(20, 5, activation='relu', input_shape=(30, 100, 3)))
Model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
Model.add(Flatten())
Model.add(Dropout(0.2))
Model.add(Dense(2, activation='tanh'))

Model.compile(optimizer="adam", loss='mean_squared_error', metrics=['accuracy'])
