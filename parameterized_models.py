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
    # Change 'a' for subgroups requiring different doses
    # Change vertical_resize_param for subgroups with different maximum treatment effects
    def __init__(self, a, resize_param=1, vertical_resize_param=1, auto_shift=True, shift_param=0):
        super().__init__()
        self.a = a
        self.shift_param = shift_param
        self.resize_param = resize_param
        self.vertical_resize_param = vertical_resize_param

        if auto_shift:
            self.shift_param = self.get_shift_param()

    def get_toxicity(self, dose_label):
        a = self.a
        c = self.vertical_resize_param
        m = self.resize_param
        b = self.shift_param
        x = dose_label

        return c * (((np.tanh(m * x + b) + 1.) / 2.) ** a)
        #return self.vertical_resize_param * (((np.tanh(self.resize_param * dose_label - self.shift_param) + 1.) / 2.) ** self.a)

    def initialize_dose_label(self, dose_skeleton):
        a = self.a
        c = self.vertical_resize_param
        m = self.resize_param
        b = self.shift_param
        y = dose_skeleton

        inner = 2. * ((y / c) ** (1. / a))
        num = np.arctanh(inner - 1.) - b
        x = num / m
        return x
        # x = ((2. / self.vertical_resize_param)  * (dose_skeleton ** (1. / self.a)))
        # return (np.arctanh(x - 1.) + self.shift_param) / self.resize_param
        #return (1./2. * np.log((1. + x)/(1. - x)) + self.shift_param) / self.resize_param

    def get_shift_param(self):
        '''
        Get shift param such that x=0 when y=0.05
        c = vertical_resize_param
        '''
        a = self.a
        c = self.vertical_resize_param
        return np.arctanh(2. * ((0.05 / c) ** (1. / a)) - 1.)


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

