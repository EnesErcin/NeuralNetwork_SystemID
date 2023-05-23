classdef RandomPlant

    properties
        prev_val
    end

    methods
        function self = RandomPlant(strt_val)
            self.prev_val = strt_val;
        end

        function [self,res] = proc(self,new_u)
            res = 0.998*self.prev_val + 0.232*new_u;
            self.prev_val = res;
        end
    end
end