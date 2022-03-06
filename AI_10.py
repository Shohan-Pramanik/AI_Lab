from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop

def main():
    model = build_model()
    
def build_model():    
	inputs = Input((28,28,1))
	x = Conv2D(filters = 4, kernel_size = (2,2), strides = (2,2), padding = 'valid')(inputs)
	x = Conv2D(filters = 4, kernel_size = (2,2), strides = (2,2), padding = 'valid')(x)
	x = Conv2D(filters = 4, kernel_size = (2,2), strides = (2,2), padding = 'valid')(x)
	x = Conv2D(filters = 4, kernel_size = (2,2), strides = (2,2), padding = 'valid')(x)
	x = Flatten()(x)
	outputs = Dense(3)(x)

	model = Model(inputs,outputs)
	model.summary()

	model.compile(loss = 'mse', optimizer = 'rmsprop')
	return model
	
if __name__ == '__main__':
    main()
