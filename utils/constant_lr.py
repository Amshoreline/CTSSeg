import os


class ConstantLR(object):

    def __init__(self, optimizer, **kwargs):
        pass

    def step(self):
        pass

    def state_dict(self):
        return 'This is ConstantLR, I will not change your learning rate'

    def load_state_dict(self, state_dict):
        pass
