from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import load_model

class TransferedModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Load the pre-trained facial expression model
        base_model = load_model('results/basic_model_15_epochs_timestamp_1740075008.keras')

        # Remove the output layer (softmax layer)
        base_model = Sequential(base_model.layers[:-1])
        
        # Freeze all layers in the base model
        for layer in base_model.layers:
            layer.trainable = False
            
        # Create a new model
        self.model = Sequential([
            # Add the pre-trained model (without its output layer)
            base_model,
            
            # Add new classification layers
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(categories_count, activation='softmax')
        ])
    
    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
