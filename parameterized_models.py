import numpy as np
import pandas as pd


class ParameterizedModel:
    def __init__(self, **args):
        pass

    def get_toxicity(self, dose_label):
        raise NotImplementedError

    def initialize_dose_label(self, dose_skeleton):
        raise NotImplementedError

class OQuiqleyModel(ParameterizedModel):
    def __init__(self, a, shift_param=0, resize_param=1):
        super().__init__()
        self.a = a
        self.shift_param = shift_param
        self.resize_param = resize_param

    def get_toxicity(self, dose_label):
        return self.resize_param * (((np.tanh(dose_label - self.shift_param) + 1.) / 2.) ** self.a)

    def initialize_dose_label(self, dose_skeleton):
        x = ((2. / self.resize_param) * (dose_skeleton ** (1. / self.a)) - 1.)
        return (1./2. * np.log((1. + x)/(1. - x)) + self.shift_param)


class SigmoidalModel(ParameterizedModel):
    def __init__(self, a):
        super().__init__()
        self.a = a

    def get_toxicity(self, dose_label):
        return 1. / (1. + np.exp(-dose_label))

    def initialize_dose_label(self, dose_skeleton):
        return np.log(dose_skeleton / (1. - dose_skeleton))

class PowerModel(ParameterizedModel):
    def __init__(self, a):
        super().__init__()
        self.a = a

    def get_toxicity(self, dose_label):
        return dose_label ** np.exp(self.a)

    def initialize_dose_label(self, dose_skeleton):
        return dose_skeleton ** (1. / np.exp(self.a)) 

class OneParamLogisticModel(ParameterizedModel):
    def __init__(self, b):
        super().__init__()
        self.b = b

    def get_toxicity(self, dose_label):
        return (np.exp(3. + dose_label * np.exp(self.b)))\
                / (1 + np.exp(3. + dose_label * np.exp(self.b)))

    def initialize_dose_label(self, dose_skeleton):
        return (np.log(dose_skeleton / (1. - dose_skeleton)) - 3.) / np.exp(self.b)

class TwoParamLogisticModel(ParameterizedModel):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def get_toxicity(self, dose_label):
        return (np.exp(self.a + dose_label * np.exp(self.b)))\
                / (1 + np.exp(self.a + dose_label * np.exp(self.b)))

    def initialize_dose_label(self, dose_skeleton):
        return (np.log(dose_skeleton / (1. - dose_skeleton)) - self.a) / np.exp(self.b)

