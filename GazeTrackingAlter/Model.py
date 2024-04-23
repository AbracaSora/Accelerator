from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

Model = Sequential(name="Accelerator")

Model.add(Input(shape=(25, 50, 3)))
Model.add(Conv2D(20, 5, activation='relu'))
Model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
Model.add(Flatten())
Model.add(Dropout(0.2))
Model.add(Dense(2, activation='tanh'))

Model.compile(optimizer="adam", loss='mean_squared_error', metrics=['accuracy'])

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
