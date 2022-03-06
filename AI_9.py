from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop


def main():
    model = build_model()

def build_model():
    inputs = Input((28, 28))
    x = Flatten()(inputs)
    x = Dense(32, activation = 'sigmoid')(x)
    x = Dense(64, activation = 'sigmoid')(x)
    x = Dense(32, activation = 'sigmoid')(x)
    x = Dense(16, activation = 'sigmoid')(x)
    outputs = Dense(10)(x)
    
    model = Model(inputs, outputs)
    model.summary()
    
    model.compile(loss = 'mse', optimizer = 'rmsprop')
    return model

if __name__ == '__main__':
    main()
