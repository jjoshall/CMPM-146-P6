from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.optimizers import RMSprop, Adam

class RandomModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Create a new model with the same architecture but random weights
        self.model = Sequential([
            layers.Rescaling(1./255, input_shape=input_shape),
            
            layers.MaxPooling2D(2),
            
            # 1st Convolutional Layer
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),
            
            # 2nd Convolutional Layer
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),
            
            # 3rd Convolutional Layer
            layers.Conv2D(48, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),
            
            # Dropout to stop overfitting
            layers.Dropout(0.25),
            
            # Flatten the results to feed into a DNN
            layers.Flatten(),
            
            # Dense Layers for the new classification task
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.25),
            layers.Dense(categories_count, activation='softmax')
        ])
    
    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    @staticmethod
    def _randomize_layers(model):
        # Your code goes here

        # you can write a function here to set the weights to a random value
        # use this function in _define_model to randomize the weights of your loaded model
        pass
