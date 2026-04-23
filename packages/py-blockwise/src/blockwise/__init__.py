"""blockwise: Blockwise Reduced Modeling (BRM) for tabular data.

References
----------
Srinivasan, K., Currim, F., Ram, S. (2025). A Reduced Modeling Approach for
Making Predictions With Incomplete Data Having Blockwise Missing Patterns.
*INFORMS Journal on Data Science*.
"""

from .brm import BRM
from .blocks import choose_num_blocks
from .simulate import simulate_blockwise_missing
from . import datasets

__version__ = "0.1.0"
__all__ = [
    "BRM",
    "choose_num_blocks",
    "simulate_blockwise_missing",
    "datasets",
    "__version__",
]
