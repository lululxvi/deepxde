import sys
from pprint import pformat

import brainunit as u
import jax.tree

from deepxde.experimental.utils import tree_repr


class TrainingDisplay:
    """
    Display training progress.
    """

    def __init__(self):
        self.len_train = None
        self.len_test = None
        self.len_metric = None
        self.is_header_print = False

    def print_one(self, s1, s2, s3, s4):
        s1 = s1.split("\n")
        s2 = s2.split("\n")
        s3 = s3.split("\n")
        s4 = s4.split("\n")

        lines = []
        for i in range(max([len(s1), len(s2), len(s3), len(s4)])):
            s1_ = s1[i] if i < len(s1) else ""
            s2_ = s2[i] if i < len(s2) else ""
            s3_ = s3[i] if i < len(s3) else ""
            s4_ = s4[i] if i < len(s4) else ""
            lines.append(
                "{:{l1}s}{:{l2}s}{:{l3}s}{:{l4}s}".format(
                    s1_,
                    s2_,
                    s3_,
                    s4_,
                    l1=10,
                    l2=self.len_train,
                    l3=self.len_test,
                    l4=self.len_metric,
                )
            )

        print("\n".join(lines))
        sys.stdout.flush()

    def header(self):
        self.print_one("Step", "Train loss", "Test loss", "Test metric")
        self.is_header_print = True

    def __call__(self, train_state):
        train_loss_repr = pformat(train_state.loss_train, width=40)
        test_loss_repr = pformat(train_state.loss_test, width=40)
        test_metrics_repr = pformat(train_state.metrics_test, width=40)

        if not self.is_header_print:
            train_loss_repr_max = max(
                [len(line) for line in train_loss_repr.split("\n") if line]
            )
            test_loss_repr_max = max(
                [len(line) for line in test_loss_repr.split("\n") if line]
            )
            test_metrics_repr_max = max(
                [len(line) for line in test_metrics_repr.split("\n") if line]
            )
            self.len_train = train_loss_repr_max + 10
            self.len_test = test_loss_repr_max + 10
            self.len_metric = test_metrics_repr_max + 10
            self.header()

        self.print_one(
            str(train_state.step),
            train_loss_repr,
            test_loss_repr,
            test_metrics_repr,
        )

    def summary(self, train_state):
        print("Best trainer at step {}:".format(train_state.best_step))
        print("  train loss: {}".format(train_state.best_loss_train))
        print("  test loss: {}".format(train_state.best_loss_test))
        print("  test metric: {}".format(tree_repr(train_state.best_metrics)))
        if train_state.best_ystd is not None:
            print("  Uncertainty:")
            print(
                "    l2: {}".format(
                    jax.tree.map(lambda x: u.linalg.norm(x), train_state.best_ystd)
                )
            )
            print(
                "    l_infinity: {}".format(
                    jax.tree_map(
                        lambda x: u.linalg.norm(x, ord=u.math.inf),
                        train_state.best_ystd,
                        is_leaf=u.math.is_quantity,
                    )
                )
            )
            if len(train_state.best_ystd) == 1:
                index = u.math.argmax(tuple(train_state.best_ystd.values())[0])
                print(
                    "    max uncertainty location:",
                    jax.tree_map(
                        lambda test: test[index],
                        train_state.X_test,
                        is_leaf=u.math.is_quantity,
                    ),
                )
        print("")
        self.is_header_print = False


training_display = TrainingDisplay()
