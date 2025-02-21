from preprocess import get_transfer_datasets
from models.transfered_model import TransferedModel
from models.random_model import RandomModel
from config import image_size, categories
import matplotlib.pyplot as plt
import numpy as np
import time

# Your code should change these values based on your choice of dataset for the transfer task
# -------------
input_shape = (image_size[0], image_size[1], 3)
categories_count = 2
# -------------

models = {
    'random_model': RandomModel,
    'transfered_model': TransferedModel,
    
}

def plot_history_diff(initial_hist, transfered_hist):
    val_acc_initial = initial_hist.history['val_accuracy']
    val_acc_tranfered = transfered_hist.history['val_accuracy']

    epochs = range(1, len(val_acc_initial) + 1)
    
    plt.figure(figsize = (24, 6))
    plt.plot(epochs, val_acc_initial, 'b', label = 'Without Transfer Learning')
    plt.plot(epochs, val_acc_tranfered, 'r', label = 'With Transfer Learning')
    plt.grid(True)
    plt.legend()
    plt.title('Far Transfer from Facial Recognition to Leukemia Classification')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.savefig('transfer_learning_comparison.png')
    plt.show()

if __name__ == "__main__":
    # Your code should change the number of epochs
    epochs = 15
    print('* Data preprocessing')
    train_dataset, validation_dataset, test_dataset = get_transfer_datasets()
    histories = []
    results = {}
    for name, model_class in models.items():
        print('* Training {} for {} epochs'.format(name, epochs))
        model = model_class(input_shape, categories_count)
        model.print_summary()
        history = model.train_model(train_dataset, validation_dataset, epochs)
        histories.append(history)
        print('* Evaluating {}'.format(name))
        model.evaluate(test_dataset)
        print('* Confusion Matrix for {}'.format(name))
        print(model.get_confusion_matrix(test_dataset))
    assert len(histories) == 2, "The number of trained models is not equal to two"
    plot_history_diff(*histories)
