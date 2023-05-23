function y = increasingStepFunction(t,sr,endtime)
    step_dur = 10;      % Duration of each step
    step_increment = 30;     % Step increment value
    y = zeros(size(t));
    acc = 0;
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
