from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model
        
        self.model = Sequential([
            Rescaling(1./255, input_shape=input_shape),

            layers.MaxPooling2D(2),

            # 1st Convolutional Layer
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),

            # 2nd Convolutional Layer
            layers.Conv2D(48, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),

            # 3rd Convolutional Layer
            layers.Conv2D(48, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),

            # Flatten the results to feed into a DNN
            layers.Flatten(),

            # Dense Layer
            layers.Dense(25, activation='relu'),
            layers.Dense(categories_count, activation='softmax')
        ])
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )