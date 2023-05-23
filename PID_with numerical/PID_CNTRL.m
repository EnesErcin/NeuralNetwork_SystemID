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
            self.Kp = 0;  self.Kd = 0; self.Ki = 0;
        end

        function result = proc(self,new_e)
            intg_res = Intg.intg(new_e);
            dif_res = Diff.diff(new_e);
            res = self.Kd*dif_res + self.Ki*intg_res + self.Kp*new_val;
            result = res;
        end
    end
end