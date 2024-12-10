"""
HONOUR CODE:
This project is an adaptation of my previous project I did for the seminar
Statistical Language Processing at the University of Tübingen in the Summer
Semester 2024. The code here is a result of my own work on the patterns given
by my lecturers (they include method declarations, docstring descriptions
and marked parts of code) and my later work to adapt the project, including
adaptation for the command line use.

Szymon T. Kossowski
"""

import argparse
import torch
from sklearn.metrics import classification_report

from trainer import FeedForwardNet
from constants import *


class Evaluator:
    def __init__(self):
        """
        Class for loading a Feedforward Network from file, and evaluating
        on test data also loaded from file.

        Public methods include:
            - load_model()
            - load_data()
            - evaluate()
        """
        self.model = None
        self.X_test = None
        self.y_test = None
        self.label_map = None

    def load_model(self, model_file):
        """
        Loads the model in model_file and saves it in self.model.

        - Load model_file, which contains a dictionary
          as saved with Trainer.save_best_model()
        - Instantiate the model
        - Set model to evaluation mode.

        See https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

        :param model_file: file containing the model
        """
        checkpoint = torch.load(model_file)
        input_size = checkpoint[N_DIMS_KEY]
        hidden_size = checkpoint[HIDDEN_SIZE_KEY]
        output_size = checkpoint[N_CLASSES_KEY]
        self.model = FeedForwardNet(input_size, hidden_size, output_size)
        self.model.load_state_dict(checkpoint[MODEL_STATE_KEY])
        self.model.eval()

    def load_data(self, data_file):
        """
        Load the evaluation tensors in data_file, where data_file
        is as generated by Preprocessor.save_tensors().
        Set self.X_test, self.y_test, and self.label_map.

        Note: You can use torch.load() here.

        :param data_file: file containing test data as tensors
        """
        data = torch.load(data_file)
        self.X_test = data[X_KEY]
        self.y_test = data[Y_KEY]
        self.label_map = {v: k for k, v in data[MAP_KEY].items()}

    def evaluate_model(self):
        """
        Evaluate the model loaded in load_model using the evaluation
        data loaded by load_data().
        Return two reports generated by the
        sklearn.metrics.classification_report package:

        - report dictionary
        - report as a string

        You will need the label_map to generate the *target_names* for the
        classification_report() function.

        Note: It is important that gradient calculation is turned off during
        evaluation. This can be done by getting the model predictions
        in a **with torch.no_grad():** block.

        :return dict, str: evaluation report as dictionary and as string
        """
        with torch.no_grad():
            outputs = self.model(self.X_test)
            _, predictions = torch.max(outputs, 1)

        target_names = [self.label_map[i] for i in range(len(self.label_map))]
        report_dict = classification_report(self.y_test, predictions, target_names=target_names, output_dict=True,
                                            zero_division=1)
        report_str = classification_report(self.y_test, predictions, target_names=target_names, zero_division=1)

        return report_dict, report_str


def parseargs():
    ap = argparse.ArgumentParser(description="Evaluate trained model on specified data.")
    ap.add_argument("--model", type=str, required=True, help="Path to the .pt model file.")
    ap.add_argument("--data", type=str, required=True, help="Path to the .pt data (dev, test or training)"
                                                            " file.")
    return ap.parse_args()


def main(args):
    evaluator = Evaluator()
    evaluator.load_model(args.model)
    evaluator.load_data(args.data)
    report_dict, report_str = evaluator.evaluate_model()
    print("Classification Report (String):\n", report_str)
    print("\nClassification Report (Dictionary):\n", report_dict)


if __name__ == '__main__':
    """
    Evaluate the following:
    
    - baseline model on the dev data
    - baseline model on the test data
    - Your best model, generated by grid search, on dev data
    - Your best model, generated by grid search, on test data
    """
    main(parseargs())
    # evaluator_dev = Evaluator()
    # evaluator_dev.load_model("Models/baseline-model-given.pt")
    # evaluator_dev.load_data("Data/dev-tensors.pt")
    # dict_dev, str_dev = evaluator_dev.evaluate_model()
    # print(dict_dev, str_dev)
    #
    # evaluator_test = Evaluator()
    # evaluator_test.load_model("Models/baseline-model-given.pt")
    # evaluator_test.load_data("Data/test-tensors.pt")
    # dict_test, str_test = evaluator_test.evaluate_model()
    # print(dict_test, str_test)
    #
    # evaluator_best_dev = Evaluator()
    # evaluator_best_dev.load_model("Models/best.pt")
    # evaluator_best_dev.load_data("Data/dev-tensors.pt")
    # dict_best_dev, str_best_dev = evaluator_best_dev.evaluate_model()
    # print(dict_best_dev, str_best_dev)
    #
    # evaluator_best_test = Evaluator()
    # evaluator_best_test.load_model("Models/best.pt")
    # evaluator_best_test.load_data("Data/test-tensors.pt")
    # dict_best_test, str_best_test = evaluator_best_test.evaluate_model()
    # print(dict_best_test, str_best_test)
