classdef Intg
    properties
        prev_val;
        sample_rate;
        cntr;
        acc;
    end

    methods
        function obj = Intg(strt_val,sample_Rate)
            obj.sample_rate = sample_Rate;
            obj.prev_val = strt_val;
            obj.cntr = 0;
            obj.acc = 0;
        end

        function [obj,intg] = intg(obj,new_val)
            obj.cntr = obj.cntr + 1;
            obj.acc = obj.acc + (obj.sample_rate*0.5)*(obj.prev_val + new_val);
            obj.prev_val = new_val;
            intg = obj.acc;
        end
    end
end