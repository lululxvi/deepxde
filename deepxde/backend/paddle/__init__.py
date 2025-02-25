import os

from .tensor import *  # pylint: disable=redefined-builtin

# enable prim if specified
enable_prim_value = os.getenv("PRIM")
enable_prim = enable_prim_value.lower() in ['1', 'true', 'yes', 'on'] if enable_prim_value else False
if enable_prim:
    # Mostly for compiler running with dy2st.
    from paddle.framework import core

    core.set_prim_eager_enabled(True)
    # The following protected member access is required.
    # There is no alternative public API available now.
    # pylint: disable=protected-access
    core._set_prim_all_enabled(True)
    print("Prim mode is enabled.")
