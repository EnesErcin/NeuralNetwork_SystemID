classdef Neuron_for_sinn
   properties
    input_num               ;
    mu              = 0.001 ;   %Learning Rate
    loss                    ;
    w                       ;   %Weigths Initited close to 1
    b                       ;
    a_cache                 ;
    y_cache                 ;
    d_cache                 ;
   end
   methods
       
      function obj = Neuron_for_sinn(num)
            % Constructor
            obj.input_num = num;
            %   Initate weigth and bias
            obj.w  = 0.001*randn([num,1]); %Weigths Initited close to 
            obj.b  = 0.001*randn;
            
            obj.a_cache                 =zeros([num,1]);
            obj.y_cache                 =zeros([num,1]);
            obj.d_cache                 =0;
      end
      
      function [obj] = feedforward(x,exp,obj)
            assert(len(x)==self.input_num,"Input number should be equal to input_num %d",self.input_num);
            obj.a_cache =   dot(obj.w,x) + obj.b; %Linear Layer
            obj.y_cache =   sigmoid(a);
            obj.d_cache =   exp;
            temp = 0;
            for i = 1:obj.input_num
                temp = temp + (exp-obj.y_cache(i))^2;
            end
            obj.loss =     temp;
      end
      function obj = backprop(obj)
          dC = -2*(obj.d_cache-obj.y_cache); %Cost Function -> dC/dY
          dy = obj.y_cache*(1-obj.y_cache);  %Y -> dY/da
          da = obj.a_cache;                  %a -> da/dw
          dw = da*dy*dC;                     %dw -> dC/dw -> dC/dY*dY/da*da/dw
            
          db = da*dy;                        %dw -> dC/db -> dC/dY*dY/da*da/db
          obj.w = obj.w -obj.mu*dw;
          obj.b = obj.b -obj.mu*db ;
      end
   end
end 

