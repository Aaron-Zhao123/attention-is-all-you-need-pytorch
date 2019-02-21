from overriders.base import CustomWrapperBase

class Pruner(CustomWrapperBase):
    def wrapper(self, value):
        return value * 0

