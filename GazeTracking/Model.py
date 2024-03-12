from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

Model = Sequential()

Model.add(Conv2D(20, 5, activation='relu', input_shape=(50, 25, 3)))
Model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
Model.add(Flatten())
Model.add(Dropout(0.2))
Model.add(Dense(2, activation='tanh'))

Model.compile(optimizer=Adam(lr=0.0005), loss='mean_squared_error', metrics=['accuracy'])
