classdef Diff
    properties
        prev_val;
        sample_rate;
        cntr;
    end

    methods
        function obj = Diff(strt_val,sample_Rate)
            obj.sample_rate = sample_Rate;
            obj.prev_val = strt_val;
            obj.cntr = 0;
        end

        function [obj,diff] = diff(obj,new_val)
            obj.cntr = obj.cntr + 1;
            acc = (new_val - obj.prev_val)/(obj.sample_rate);
            obj.prev_val = new_val;
            diff = acc;
        end
    end
end