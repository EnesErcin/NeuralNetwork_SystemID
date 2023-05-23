classdef PID_CNTRL
    properties
        Intg_calc
        Diff_calc
        Kp
        Ki
        Kd
    end

    methods
        function self = PID_CNTRL(strt_val,sr)
            self.Intg_calc = Intg(strt_val,sr);
            self.Diff_calc = Diff(strt_val,sr);
            self.Kp = 0.9;  self.Kd = 0.4; self.Ki = 0.2;
        end

        function [self,result] = proc(self,new_e)
            [self.Intg_calc, intg_res] = self.Intg_calc.intg(new_e);
            [self.Diff_calc, dif_res] = self.Diff_calc.diff(new_e);
            res = self.Kd*dif_res + self.Ki*intg_res + self.Kp*new_e;
            result = res;
        end
    end
end