from numeric_intg import Intg_num
from numeric_dif  import Diff_num

class PID_CNTRL():
    def __init__(self,Kp = 0.9,Kd = 0.4 ,Ki = 0.2 ):
        self.Kp = Kp 
        self.Ki = Ki
        self.Kd = Kd

        self.INTG_BLOCK = Intg_num(1,0)
        self.Diff_num = Diff_num(1,0)

    def _proc(self,new_e):
        intg_res = self.INTG_BLOCK(new_e)
        dif_res  = self.Diff_num(new_e)

        res = self.Kd*dif_res + self.Ki*intg_res + self.Kp*new_e
        
        return res