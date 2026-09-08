"""State-only C1 coordinator for the frozen ms18 executor.

Keep package import torch-free: Isaac Gym requires itself to be imported before
PyTorch in closed-loop entry points.  Concrete classes live in their modules.
"""
