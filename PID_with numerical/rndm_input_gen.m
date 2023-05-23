function y = rndm_input_gen(t,sr,endtime,strt)
    step_dur = 60;      % Duration of each step
    step_increment = 20;     % Step increment value
    y = zeros(size(t));
    acc = strt;
    cnt = 0;

    for t = 1:sr:endtime
        y(t) = acc;
        cnt = cnt + 1;
        if cnt >= step_dur
            cnt = 0;
            acc = acc + step_increment;
        end
    end
end
