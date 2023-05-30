class Plant():
    def __init__(self,prev_val = 0):
        self.prev_val = prev_val
    
    def _proc(self,new_u):
        res = 0.998*self.prev_val + 0.232*new_u
        self.prev_val = res
        return res