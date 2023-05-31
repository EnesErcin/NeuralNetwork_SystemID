import numpy as np


class NN_PID_interface():
    def __init__(self,is_linear) -> None:
        self.prev_error = np.zeros((3,1))
        self.is_linear = is_linear
        assert(type(is_linear) == bool)

        if is_linear:
            self.sys_coef = 0.988
        else:
            self.sys_coef = 0.9

    
    def _proc(self,new_e):
        dyn1_dyn = self.sys_coef
        dJ_dyn1 = -2*new_e

        dJ_dyn = dJ_dyn1*dyn1_dyn

        dJ_dk = dJ_dyn*self.prev_error 
        self.prev_error = new_e*np.ones((3,1))

        return dJ_dk
