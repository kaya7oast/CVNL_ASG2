"""Training module"""

from .CNN_trainer import train_epoch, validate, CNN_train_two_phase

__all__ = ['train_epoch', 'validate', 'CNN_train_two_phase']
