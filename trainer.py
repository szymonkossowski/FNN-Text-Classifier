"""
Course:        Statistical Language Processing - Summer 2024
Assignment:    (A2)
Author(s):     (Szymon Tomasz Jarogniew Kossowski)

Honor Code:    I pledge that this program represents my own work,
               and that I have not given or received unauthorized help
               with this assignment.
"""

import torch
import torch.nn as nn
import copy
import json
from sklearn.metrics import f1_score
import argparse

import constants
from constants import *


# Class FeedForwardNet(nn.Module) by my lecturers
class FeedForwardNet(nn.Module):
    def __init__(self, n_dims, hidden_size, n_classes):
        """
        A feedforward network for multi-class classification with 1 input layer,
        1 hidden layer, and 1 output layer.

        :param n_dims: number of dimensions in input
        :param hidden_size: number of neurons in the hidden layer
        :param n_classes: number of classes
        """
        super(FeedForwardNet, self).__init__()
        # model layers
        self.linear1 = nn.Linear(n_dims, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        """
        The forward pass applies the network layers to x.

        :param x: the input data as a tensor of size (n_samples, embedding_dim)
        :return: the output of the last layer
        """
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


class Trainer:
    # Init function by my lecturers
    def __init__(self):
        """
        Class for training a Feedforward Network.

        Public methods include:
            - load_data()
            - train()
            - save_best_model()
        """

        # class variables that will be set later
        self.X_train = None  # train X tensors
        self.y_train = None  # train y tensors
        self.X_dev = None  # dev X tensors
        self.y_dev = None  # dev y tensors
        self.label_map = None  # {label:class_code} dictionary
        self.n_dims = None  # number of dimensions in training data
        self.n_classes = None  # number of classes in training data
        self.best_model = None

    def _load_train_tensors(self, train_tensor_file):
        """
        Private method to load the training tensors in train_tensor_file
        into class variables self.X_train, self.y_train, and self.label_map.
        Assumes that the file uses keys X_KEY, Y_KEY, MAP_KEY,
        as defined in constants.py.
        Helper method for load_data().

        Note: You can use torch.load() here.

        :param train_tensor_file: file containing training tensors
        """
        # loading train tensors
        data = torch.load(train_tensor_file)
        self.X_train = data[X_KEY]
        self.y_train = data[Y_KEY]
        self.label_map = data[MAP_KEY]

    def _load_dev_tensors(self, dev_tensor_file):
        """
        Private method to load the dev tensors in dev_tensor_file
        into class variables self.X_dev, self.y_dev, and self.label_map.
        Assumes that the file uses keys X_KEY, Y_KEY, MAP_KEY,
        as defined in constants.py.
        Helper method for load_data().

        Note: You can use torch.load() here.

        :param dev_tensor_file: file containing dev tensors
        """
        # loading dev tensors
        data = torch.load(dev_tensor_file)
        self.X_dev = data[X_KEY]
        self.y_dev = data[Y_KEY]
        self.label_map = data[MAP_KEY]

    def load_data(self, train_tensor_file, dev_tensor_file):
        """
        Public method to load train and dev tensors from files,
        as saved in the Preprocessor class.

        Also sets self.n_dims, and self.n_classes.

        :param train_tensor_file: file containing training tensors
        :param dev_tensor_file: file containing dev tensors
        """
        # loading all the necessary data
        self._load_train_tensors(train_tensor_file)
        self._load_dev_tensors(dev_tensor_file)
        self.n_dims = self.X_train.shape[1] if self.X_train is not None else None
        self.n_classes = len(self.label_map)

    def _macro_f1(self, model):
        """
        Private method to calculate the macro f1 score of the given model
        on the dev data.
        Helper method for _training_loop().

        Note that the predictions on the dev data is the output of the forward pass,
        with shape (n_samples, n_classes).
        This means that you need to get the index of the highest value in each
        row of predictions (which is the class code of the predicted class).
        You can use torch.argmax() for that.
        Use sklearn.metrics.f1_score to calculate the macro-averaged F1 score.

        Note: It is important that gradient calculation is turned off here,
        which can be done by putting the code for this function in
        a **with torch.no_grad():** block.

        :param model: the model to test on the dev data
        :return: float - macro F1 score
        """
        # calculating the macro f1 from the highest values in each row of predictions
        # Content of the line 148 by my lecturers
        with torch.no_grad():
            predictions = model(self.X_dev)
            predicted_labels = torch.argmax(predictions, 1)
            f1 = f1_score(self.y_dev.cpu(), predicted_labels.cpu(), average='macro')
        return f1

    def _training_loop(self, model, loss_fn, optimizer, n_epochs):
        """
        This is where the actual training takes place.
        Private method to train model using the given loss function,
        optimizer, and n_epochs.
        Helper method for train().

        Training and dev data are stored as class variables.

        At each epoch, evaluate the model on the dev data.
        If the macro-averaged F1 score is better than the current best score,
        update the current best score, best epoch, and best model (you must make
        a deep copy of the model state).

        Returns a dictionary containing information about the best model (the one
        with the highest macro-averaged F1 score, not the last model).
        The returned dictionary should contain the following keys:

        - MODEL_STATE_KEY: make a deep copy: copy.deepcopy(model.state_dict())
        - F1_MACRO_KEY: F1 score of the best model
        - BEST_EPOCH_KEY: epoch of the best model

        :param model: the model to train
        :param loss_fn: the loss function
        :param optimizer: the optimizer
        :param n_epochs: number of training epochs
        :return: dictionary containing model state, F1 score, and epoch of the best model
        """
        best_f1 = 0.0
        best_epoch = 0
        best_model_state = None

        for epoch in range(n_epochs):
            # training loop
            model.train()
            optimizer.zero_grad()
            outputs = model(self.X_train)
            loss = loss_fn(outputs, self.y_train)
            loss.backward()
            optimizer.step()

            # evaluating on dev data
            model.eval()
            with torch.no_grad():
                f1 = self._macro_f1(model)

            # checking if this is the best model
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())

        # saving the result dictionary
        result = {
            MODEL_STATE_KEY: best_model_state,
            F1_MACRO_KEY: best_f1,
            BEST_EPOCH_KEY: best_epoch
        }
        return result

    def train(self, hidden_size, n_epochs, learning_rate):
        """
        Public method to train a model.

        - Create a model (an instance of FeedForwardNet). The parameters of the
          model are stored in class variables, and hyperparameters are passed in
          to this method.
        - Set the loss function (CrossEntropyLoss) and optimizer (AdamW)
        - Train the model and add the following keys to the dictionary returned
          by _training_loop():
            - HIDDEN_SIZE_KEY
            - N_DIMS_KEY
            - N_CLASSES_KEY
            - LEARNING_RATE_KEY
            - N_EPOCHS_KEY
            - OPTIMIZER_NAME_KEY get with optimizer.__class__.__name__
            - LOSS_FN_NAME_KEY get with loss_fn.__class__.__name__
        - Store the updated dictionary in self.best_model
        - Return updated dictionary

        :param n_epochs: number of epochs
        :param hidden_size: hidden_size
        :param learning_rate: learning rate
        :return: best model dictionary containing the model state and all metadata
        """

        # Content of the line 243 by my lecturers
        # Use a seed to make sure that results are reproducible.
        # Please do not remove or change the seed.
        torch.manual_seed(42)

        # creating a model
        model = FeedForwardNet(self.n_dims, hidden_size, self.n_classes)

        # setting up loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # training the model
        result = self._training_loop(model, loss_fn, optimizer, n_epochs)

        # adding metadata to the result dictionary
        result[HIDDEN_SIZE_KEY] = hidden_size
        result[N_DIMS_KEY] = self.n_dims
        result[N_CLASSES_KEY] = self.n_classes
        result[LEARNING_RATE_KEY] = learning_rate
        result[N_EPOCHS_KEY] = n_epochs
        result[OPTIMIZER_NAME_KEY] = optimizer.__class__.__name__
        result[LOSS_FN_NAME_KEY] = loss_fn.__class__.__name__

        # storing the updated dictionary in self.best_model
        self.best_model = result

        # return the updated dictionary
        return result

    def save_best_model(self, base_filename):
        """
        Save the trained model in self.best_model, as well as its metadata.

        2 dictionaries are saved:

        - base_filename.pt (use torch.save())
          The model and all information required to load it:
                - MODEL_STATE_KEY
                - N_DIMS_KEY
                - N_CLASSES_KEY
                - HIDDEN_SIZE_KEY

        - base_filename-info.json (use json library)
          Metadata about the model (all keys except MODEL_STATE_KEY):

                - HIDDEN_SIZE_KEY
                - N_DIMS_KEY
                - N_CLASSES_KEY
                - LEARNING_RATE_KEY
                - N_EPOCHS_KEY
                - BEST_EPOCH_KEY
                - F1_MACRO_KEY
                - OPTIMIZER_NAME_KEY
                - LOSS_FN_NAME_KEY

        :param base_filename: path and base name to save files (e.g. "Models/best")
        """
        # Save model
        model_path = base_filename + ".pt"
        pt_info = {MODEL_STATE_KEY: self.best_model[MODEL_STATE_KEY],
                   N_DIMS_KEY: self.best_model[N_DIMS_KEY],
                   N_CLASSES_KEY: self.best_model[N_CLASSES_KEY],
                   HIDDEN_SIZE_KEY: self.best_model[HIDDEN_SIZE_KEY]}
        torch.save(pt_info, model_path)

        # Save metadata
        model_info = {
            HIDDEN_SIZE_KEY: self.best_model[HIDDEN_SIZE_KEY],
            N_DIMS_KEY: self.best_model[N_DIMS_KEY],
            N_CLASSES_KEY: self.best_model[N_CLASSES_KEY],
            LEARNING_RATE_KEY: self.best_model[LEARNING_RATE_KEY],
            N_EPOCHS_KEY: self.best_model[N_EPOCHS_KEY],
            BEST_EPOCH_KEY: self.best_model[BEST_EPOCH_KEY] + 1,
            F1_MACRO_KEY: self.best_model[F1_MACRO_KEY],
            OPTIMIZER_NAME_KEY: self.best_model[OPTIMIZER_NAME_KEY],
            LOSS_FN_NAME_KEY: self.best_model[LOSS_FN_NAME_KEY]
        }

        model_info_path = base_filename + "-info.json"
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=4)  # indentation for readability


def parseargs():
    ap = argparse.ArgumentParser(description="Train a FeedForward Network on preprocessed tensors.")
    ap.add_argument("-t", "--train_tensor_file", type=str, required=True, help="Path to the training tensors file.")
    ap.add_argument("-d", "--dev_tensor_file", type=str, required=True, help="Path to the dev tensors file.")
    ap.add_argument("-h", "--hidden_size", type=int, default=8, help="Number of neurons in the hidden layer.")
    ap.add_argument("-e", "--n_epochs", type=int, default=200, help="Number of training epochs.")
    ap.add_argument("-l", "--learning_rate", type=float, default=0.01, help="Learning rate for the optimiser.")
    ap.add_argument("-s", "--output_model", type=str, required=True, help="Base path to save the trained model and "
                                                                          "metadata.")
    return ap.parse_args()


def main(args):
    trainer = Trainer()
    trainer.load_data(args.train_tensor_file, args.dev_tensor_file)
    trainer.train(args.hidden_size, args.n_epochs, args.learning_rate)
    trainer.save_best_model(args.output_model)


if __name__ == '__main__':
    """
    Try out your Trainer here.
    If you train and save a model using the same hyperparameters as
    Models/baseline-model-given, you should get the same results.
    """
    main(parseargs())
    # trainer = Trainer()
    #
    # train_tensor_file = "Data/train-tensors.pt"
    # dev_tensor_file = "Data/dev-tensors.pt"
    #
    # trainer.load_data(train_tensor_file, dev_tensor_file)
    # trainer.train(8, 200, 0.01)
    # trainer.save_best_model("Models/best")
