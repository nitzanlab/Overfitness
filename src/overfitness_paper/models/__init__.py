"""
Function-class implementations.

Linear-map and polynomial models are exposed via the dispatcher in
`simulation.set_function_type`; they live in `simulation.py` for backward
compatibility with the original runner scripts. Neural-network sampling
and forward-pass code (used for SI Figs S2, S4, S5D-F, S6D-F) is here in
`neural_net.py`.
"""

from . import neural_net
