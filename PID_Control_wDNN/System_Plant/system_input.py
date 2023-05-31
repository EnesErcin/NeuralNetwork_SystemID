import numpy as np

def _input_gen(is_linear,t,sr,end_t,strt_val):

    if is_linear:
        step_dur  = 60
        step_inc = 20
        acc = strt_val
        cnt = 0

        y = np.zeros(((np.size(t)+1)))

        for i in range (0,end_t+1):
            y[i] = acc 

            if cnt >=step_dur:
                cnt = 0
                acc = acc + step_inc

            cnt = cnt +1
            
        return y
    
    else:
        step_dur  = 60
        fst_step_inc = -40
        snc_step_inc = 20
        acc = 60
        cnt = 0
        step_cnt = 0

        y = np.zeros(((np.size(t)+1)))

        for i in range (0,end_t+1):
            y[i] = acc 

            if step_cnt == 0:
                if cnt >=step_dur:
                    cnt = 0
                    step_cnt = step_cnt + 1
                    acc = acc + fst_step_inc

            elif step_cnt == 1:
                if cnt >=step_dur:
                    cnt = 0
                    step_cnt = step_cnt + 1
                    acc = acc + snc_step_inc

            cnt = cnt +1
            
        return y

