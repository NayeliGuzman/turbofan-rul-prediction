
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv1D, LSTM, BatchNormalization, MaxPooling1D, Reshape
from tensorflow.keras.utils import to_categorical

class CNN_LSTM(Sequential):
  def __init__(self, input_shape=(30,18)):
    super(CNN_LSTM, self).__init__()

    # Conv. block 1
    self.add(Conv1D(filters=25, kernel_size=10, padding='same', activation='elu', input_shape=input_shape))
    self.add(MaxPooling1D(pool_size=2, padding='same'))
    self.add(BatchNormalization())
    self.add(Dropout(0.5))

    # Conv. block 2
    self.add(Conv1D(filters=50, kernel_size=10, padding='same', activation='elu'))
    self.add(MaxPooling1D(pool_size=2, padding='same'))
    self.add(BatchNormalization())
    self.add(Dropout(0.5))

    # Conv. block 3
    self.add(Conv1D(filters=100, kernel_size=10, padding='same', activation='elu'))
    self.add(MaxPooling1D(pool_size=2, padding='same'))
    self.add(BatchNormalization())
    self.add(Dropout(0.5))

    # Conv. block 4
    self.add(Conv1D(filters=200, kernel_size=10, padding='same', activation='elu'))
    self.add(MaxPooling1D(pool_size=2, padding='same'))
    self.add(BatchNormalization())
    self.add(Dropout(0.5))

    # FC+LSTM layers
    self.add(Flatten()) # Adding a flattening operation to the output of CNN block
    self.add(Dense((100))) # FC layer with 100 units
    self.add(Reshape((100,1))) # Reshape my output of FC layer so that it's compatible
    self.add(LSTM(10, dropout=0.6, recurrent_dropout=0.1, return_sequences=False))


    # Output layer
    self.add(Dense(1)) # Output FC
