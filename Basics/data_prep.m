function [fulldata,preped] = data_prep(type,input_num,sample_freq,test_size)
    if type == "sin"
        inc = 2*pi/sample_freq;
          
        a = 0:inc:2*pi;
        i = 1;
        while size(a) < test_size
            i = i+1;
            a = 0:inc:2*pi*i;
        end
     
        fulldata =sin(a);
        temp = zeros(input_num,test_size);
       
        for i = 1:(test_size-input_num)
            for j = 1:input_num
             temp(j,i) = fulldata(1,j+i).';
            end
        end 
        preped = temp;
    end
    if 0
         fprintf("at i,j = (%d,%d) \t %d \n",i,j,data(1,j+i).');
    end
end
