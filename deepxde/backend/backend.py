"""This file defines the unified tensor framework interface required by DeepXDE.

The principles of this interface:
* There should be as few interfaces as possible.
* The interface is used by DeepXDE system so it is more important to have
  clean definition rather than convenient usage.
* Default arguments should be avoided.
* Keyword or positional arguments should be avoided.
* Argument type should be easier to understand.

It is recommended the frameworks implement all the interfaces. However, it is
also OK to skip some. The generated backend module has an ``is_enabled`` function
that returns whether the interface is supported by the framework or not.
"""

# The backend currently being used
# bkd will be set to one of the following backends.
bkd = None
# All possible backends to use explicitly
tf = None
torch = None
