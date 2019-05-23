from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys


class TrainingDisplay:
    """Display training progress."""

    def __init__(self):
        self.len_train = None
        self.len_test = None
        self.len_metric = None
        self.is_header_print = False

    def clear(self):
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
            self.len_train = len(train_state.loss_train) * 8 + 8
            self.len_test = len(train_state.loss_test) * 8 + 8
            self.len_metric = len(train_state.metrics_test) * 8 + 8
            self.header()
        self.print_one(
            str(train_state.step),
            list_to_str(train_state.loss_train),
            list_to_str(train_state.loss_test),
            list_to_str(train_state.metrics_test),
        )


training_display = TrainingDisplay()


def display_best(train_state):
    print(
        "\nBest model at step {:d} with train loss {:.2e}, test loss {:.2e}, test metric {:s}".format(
            train_state.best_step,
            train_state.best_loss_train,
            train_state.best_loss_test,
            list_to_str(train_state.best_metrics),
        )
    )


def list_to_str(nums):
    return "[{:s}]".format(", ".join(["{:.2e}".format(x) for x in nums]))
