import numpy as np

def _input_gen(t,sr,end_t,strt_val):
    step_dur  = 60
    step_inc = 20
    acc = strt_val
    cnt = 0

    y = np.zeros(((np.size(t))))

    for i in range (0,end_t):
        y[i] = acc 

        if cnt >=step_dur:
            cnt = 0
            acc = acc + step_inc

        cnt = cnt +1
