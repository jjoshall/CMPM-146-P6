from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.optimizers import RMSprop

class RandomModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Create a matching architecture to the transfer model
        self.model = Sequential([
            layers.Rescaling(1./255, input_shape=input_shape),
            
            # First maxpool to reduce input size
            layers.MaxPooling2D(2),
            
            # Single conv layer with same params as transfer
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            
            # Second maxpool
            layers.MaxPooling2D(2),
            
            # Matching dense layers
            layers.Dropout(0.25),
            layers.Flatten(),
            # Reduced size of dense layer to control parameters
            layers.Dense(8, activation='relu'),
            layers.Dropout(0.25),
            layers.Dense(categories_count, activation='softmax')
        ])
    
    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )