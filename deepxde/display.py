import sys

import numpy as np

from .utils import list_to_str


class TrainingDisplay:
    """Display training progress."""

    def __init__(self):
        self.len_train = None
        self.len_test = None
        self.len_metric = None
        self.is_header_print = False

    def print_one(self, s1, s2, s3, s4):
        print(
            "{:{l1}s}{:{l2}s}{:{l3}s}{:{l4}s}".format(
                s1,
                s2,
                s3,
                s4,
                l1=10,
                l2=self.len_train,
                l3=self.len_test,
                l4=self.len_metric,
            )
        )
        sys.stdout.flush()

    def header(self):
        self.print_one("Step", "Train loss", "Test loss", "Test metric")
        self.is_header_print = True

    def __call__(self, train_state):
        if not self.is_header_print:
            self.len_train = len(train_state.loss_train) * 10 + 4
            self.len_test = len(train_state.loss_test) * 10 + 4
            self.len_metric = len(train_state.metrics_test) * 10 + 4
            self.header()
        self.print_one(
            str(train_state.step),
            list_to_str(train_state.loss_train),
            list_to_str(train_state.loss_test),
            list_to_str(train_state.metrics_test),
        )

    def summary(self, train_state):
        print("Best model at step {:d}:".format(train_state.best_step))
        print("  train loss: {:.2e}".format(train_state.best_loss_train))
        print("  test loss: {:.2e}".format(train_state.best_loss_test))
        print("  test metric: {:s}".format(list_to_str(train_state.best_metrics)))
        if train_state.best_ystd is not None:
            print("  Uncertainty:")
            print("    l2: {:g}".format(np.linalg.norm(train_state.best_ystd)))
            print(
                "    l_infinity: {:g}".format(
                    np.linalg.norm(train_state.best_ystd, ord=np.inf)
                )
            )
            print(
                "    max uncertainty location:",
                train_state.X_test[np.argmax(train_state.best_ystd)],
            )
        print("")
        self.is_header_print = False


training_display = TrainingDisplay()
