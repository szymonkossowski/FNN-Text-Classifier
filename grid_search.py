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
from trainer import Trainer
from constants import *
import argparse


def parseargs():
    ap = argparse.ArgumentParser(description="Description")
    ap.add_argument("-d", "--data-dev", type=str, required=True, help="Path to the development data tensor file.")
    ap.add_argument("-t", "--data-test", type=str, required=True, help="Path to the test data tensor file.")
    return ap.parse_args()


def main(args):
    """
    Find the best combination of hyperparameters using a simple grid search.
    A few values for each hyperparameter should be enough.

    You do NOT need to save all the models trained during the grid search.
    Rather, keep track of the hyperparameters used for the current best model,
    similar to what you did in the training loop.

    Note that the macro F1 score was calculated during training, and stored
    as part of the dictionary returned by Trainer.train().
    You can get the model score from this dictionary, rather than
    recalculating the score.

    When the grid search is finished, train a model using the best hyperparameters,
    and save that one to Models/best-model (generating files Models/best-model.pt and
    Models/best-model-info.json).

    You will also want to print the scores and hyperparameters, to use in your discussion.
    """
    trainer = Trainer()
    trainer.load_data(train_tensor_file=args.data_test, dev_tensor_file=args.data_dev)

    # simple grid search
    learning_rates = [0.001, 0.005, 0.01]
    epochs = [2, 4, 8]
    hidden_sizes = [100, 150, 200]
    best_f1 = 0.0
    best_parameters = None

    # Code in lines 55-57 given by my lecturers
    for h_size in hidden_sizes:
        for n_epochs in epochs:
            for lr in learning_rates:
                print(f"Training with hidden_size={h_size}, learning_rate={lr}, epochs={n_epochs}")
                # train using current hyperparameters
                result = trainer.train(h_size, n_epochs, lr)
                # record the hyperparameters and the model's f1-score
                f1 = result[F1_MACRO_KEY]
                # keep track of best hyperparameters
                if f1 > best_f1:
                    best_f1 = f1
                    best_parameters = {
                        HIDDEN_SIZE_KEY: h_size,
                        N_EPOCHS_KEY: n_epochs,
                        LEARNING_RATE_KEY: lr
                    }

    print(f"Best hyperparameters of the model: {best_parameters}")
    # re-train using the best hyperparameters and save to Models/best-model.

    print("Training the best model with the best hyperparameters...")
    best_trainer = Trainer()
    best_trainer.load_data(train_tensor_file=args.data_test, dev_tensor_file=args.data_dev)
    best_trainer.train(best_parameters[HIDDEN_SIZE_KEY], best_parameters[N_EPOCHS_KEY],
                       best_parameters[LEARNING_RATE_KEY])
    best_trainer.save_best_model("Models/best-model")
    print("Best model saved as Models/best-model.pt")


if __name__ == '__main__':
    main(parseargs())
