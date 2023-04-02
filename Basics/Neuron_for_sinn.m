classdef Neuron_for_sinn <handle
   properties
    input_num               ;
    mu              = 0.01 ;   %Learning Rate
    loss                    ;
    w                       ;   %Weigths Initited close to 1
    b                       ;
    a_cache                 ;
    y_cache                 ;
    d_cache                 ;
    count                   ;
    input_cache             ;
    % Model Evaluation
    weigth_results          ;
    bias_results            ;
    cost_result             ;
    test_num                ;
    fileID = fopen('exp.txt','w');
   end
   methods
       
      function obj = Neuron_for_sinn(num,test_num)
            % Constructor
            obj.input_num = num;
            %   Initate weigth and bias
            obj.w  = 0.001*randn([num,1]); %Weigths Initited close to 
            obj.b  = 0.001*randn([num,1]);
            
            obj.a_cache                 =zeros([num,1]);
            obj.y_cache                 =zeros([num,1]);
            obj.d_cache                 =0;
            obj.count                   =0;
            obj.input_cache             =zeros([num,1]);
            obj.cost_result             =zeros([test_num,1]);
            obj.test_num                =test_num; 
            obj.weigth_results          =zeros([test_num,num]);
            obj.bias_results            =zeros([test_num,num]);
      end
      
      function [obj,exp] = feedforward(obj,x,exp,train)
          
            assert(length(x)==obj.input_num,"Input number should be equal to input_num %d",obj.input_num);
            obj.input_cache = x;
            obj.a_cache =   obj.w.*x + obj.b; %Linear Layer
            for i = 1:obj.input_num
                obj.y_cache(i) =   sigmoid(obj.a_cache(i));
            end
            obj.d_cache =   exp;
            temp = 0;
            for i = 1:obj.input_num
                temp = temp + 0.5*(exp-obj.y_cache(i))^2;
            end
            
            obj.loss =     temp;
            if train == 1 
                
                obj.count =   obj.count + 1;
                obj.cost_result(obj.count) = temp;
                obj.weigth_results(obj.count,:)  = obj.w;
                obj.bias_results(obj.count,:) = obj.b;
            end

      end
      function obj = backprop(obj)
          dC_dY = zeros(obj.input_num,1);
 
          for i = 1:obj.input_num
            dC_dY(i) = -2*(obj.d_cache-obj.y_cache(i)); %Cost Function -> dC/dY
          end
         
          dY_da = obj.y_cache.*(ones(obj.input_num,1)-obj.y_cache);  %Y -> dY/da
          da_dw = obj.input_cache;           %a -> da/dw
          dw = dC_dY.*dY_da.*da_dw;          %dw -> dC/dw -> dC/dY*dY/da*da/dw
          db = dC_dY.*dY_da;                 %dw -> dC/db -> dC/dY*dY/da*da/db   
          obj.w = obj.w -obj.mu*dw;
          obj.b = obj.b -obj.mu*db ;
      end
   end
end 

