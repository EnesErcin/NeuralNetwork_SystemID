import numpy as np

class Plant():
    def __init__(self,prev_val_y = 0,is_linear =True):
        self.prev_val_y = prev_val_y
        self.prev_prev_val_y = 0
        self.prev_u = 0
        self.is_linear = is_linear
        assert(type(is_linear)==bool)

    def _proc(self,new_u):

        if self.is_linear:
            res = 0.998*self.prev_val_y + 0.232*new_u
            self.prev_val_y = res
            return res
        else:
            # >> y(n+1)
            res = 0.9*self.prev_val_y - 0.001*self.prev_prev_val_y**2 + new_u + np.sin(self.prev_u )
            
            # y(n-1) = y(n)
            self.prev_prev_val_y = self.prev_val_y
            
            # y(n) = y(n+1)
            self.prev_val_y = res
            
            # u(n) = u(n-1)
            self.prev_u = new_u

            return res
