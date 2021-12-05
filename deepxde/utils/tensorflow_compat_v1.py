"""Utilities of tensorflow.compat.v1."""

from ..backend import tf


def guarantee_initialized_variables(session, var_list=None):
    """Guarantee that all the specified variables are initialized.

    If a variable is already initialized, leave it alone. Otherwise, initialize it.
    If no variables are specified, checks all variables in the default graph.

    Args:
        var_list: List of Variable objects.

    References:

    - https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables
    - https://www.programcreek.com/python/example/90525/tensorflow.report_uninitialized_variables
    """
    name_to_var = {v.op.name: v for v in tf.global_variables() + tf.local_variables()}
    uninitialized_variables = [
        name_to_var[name.decode("utf-8")]
        for name in session.run(tf.report_uninitialized_variables(var_list))
    ]
    session.run(tf.variables_initializer(uninitialized_variables))
    return uninitialized_variables
