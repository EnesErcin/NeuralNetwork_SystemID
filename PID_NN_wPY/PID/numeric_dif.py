

class Diff_num():
    def __init__(self,sample_rate=1,prev_val = 0):
        self.prev_val = prev_val
        self.sr = sample_rate
        self.cntr = 0

    def intg(self,new_val):
        self.cntr  = self.cntr + 1
        self.acc = (new_val- self.prev_val)/(self.sr)
        self.prev_val = new_val
        return self.acc
