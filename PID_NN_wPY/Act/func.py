import numpy as np

class neuron_layers:
    def __init__(self,layer_width,learning_rate):
        self.layer_width = layer_width
        self.param_check()
        self.hidden_size = len(layer_width) -2 # Include input at first || Output size at last
        

        self.layers = []
        ## Generate  >> input and  hidden layers
        for i in range (0,self.hidden_size):
            my_neuron = neuron_layer(layer_width[i],layer_width[i+1],learning_rate)
            self.layers.append(my_neuron)
        
        ## Generate >> output layer
        my_outlayer = output_layer(layer_width[-2],layer_width[-1],"mse",learning_rate)
        self.layers.append(my_outlayer)

    def param_check(self):
        ## Check the parameters
        assert type(self.layer_width) == list # Should be array

    def calc_output(self,x,exp):
        temp = self.forward(x)

        print("\n \n |||| In neuron {}".format(self.hidden_size))
        [j,a] = self.layers[-1].extract_output(temp[-1],exp)
        a = np.reshape(a,(np.size(a),1))
        temp.append(a)

        return [j,a]
    
    def forward(self,x):
        temp = []
        temp.append(x)
        
        for i in range (0,self.hidden_size):
            print("\n \n |||| In neuron {}".format(i))
            a = self.layers[i].forward(temp[i])
            y = self.layers[i].non_linear("sig",a)
            y = np.reshape(y,(np.size(y),1))
            temp.append(y)
            print("-------------------------\n")

        a = self.layers[-1].forward(temp[-1])
        print("-------------------------\n")

        return temp
    
    def back_prop(self):
        cnt = self.hidden_size -1 
        da_dc = None
        for a in range (0,self.hidden_size+1):
            cnt = self.hidden_size-a
            da_dc,dJ_dw = self.layers[cnt].backprop(da_dc)
            print("@ {} da_dc {}".format(cnt,dJ_dw))
        pass



class neuron_layer:
    def __init__(self,input_size,layer_width,lr=10**-5):

        ####### |  | |  |   |  |    |  | |  |    |  |    |  |  |  |     |  | #####
        ####### |  | |  |   |  |    |  | |  |    |  |    |  |  |  |     |  | #####
        ####### |x1|*|w1| + |b1| ,  |x1|*|w2|  + |b2| ,  |x1| *|w3| +   |b3| #####
        ####### |  | |  |   |  |    |  | |  |    |  |    |  |  |  |     |  | #####
        self.lr = lr
        self.layer_width = layer_width
        self.input_size = input_size
        self.w = np.ones((layer_width,input_size))
        self.b = np.random.rand(layer_width,1)

        ########################################
        ###### STORED VALUES FOR BACKPROP !! ###
        ########################################
        ## Store sigmoid 
        self.sig = np.zeros([layer_width,1])
        ## Store x
        self.x = np.zeros([input_size,1])

    def forward(self,x):
        self.clear()        
        print("Forward Calculations || W.x + b || == a \n")
        print("W -> {}, x -> {}, b ->{}".format(self.w,x,self.b))
        ##  a = w*x + b
        self.x = x
        a = np.dot(self.w,x)
        a = a + self.b
        print("a -> {}".format(a))
        return a
    
    def non_linear(self,fnc,a):
        ## σ(a) = y
        self.param_check(fnc,a)

        if fnc == "sig":
            sig = 1.0/(1.0 + np.exp(-a))
        self.sig = sig
        print("Nonlinear ||σ(a) = y|| == a \n")
        print("a -> {} ".format(a))
        print("σ(a) -> {}".format(self.sig))
        return sig

    def param_check(self,fnc,a):
        assert type(fnc) == str
        assert np.shape(a) == (self.layer_width,1) 
    
    def clear(self):
        self.sig = np.zeros([self.layer_width,1])
        self.x = np.zeros([self.input_size,1])

    def backprop(self,dJ_dc):
        ####
        ####  w*c_t + b = a_(t+1)
        ####  c_(t+1) = σ(a_(t+1))
        ####
        ####  da_dc(t)= w
        ####  dc(t+1)_da = -c(1-c)

        
        da_1 = -self.sig*(1-self.sig)
        
        dw = dJ_dc*da_1
        dJ_dw = dw*self.x
        self.w = self.w + self.lr*dJ_dw

        print("dw_2 = \t{} d_a1 = \t{} dJ_dc = \t{}".format(self.x, self.sig, dJ_dc))

        return dw, dJ_dw
    
class output_layer(neuron_layer):
    def __init__(self,input_size,output_num,coss_fnc,learning_rate):
        neuron_layer.__init__(self,input_size,output_num,learning_rate)
        self.coss_fnc = coss_fnc
        ## Store error
        self.j = np.zeros([self.layer_width,1])
    
    def non_linear(self,*args):
        # Output layer does not have activatison function in this case
        pass 

    def clear(self):
        self.j = np.zeros([self.layer_width,1])
        self.x = np.zeros([self.input_size,1])
    
    
    def backprop(self,*args):

        ## J = (r-y)**2
        ## dJ/dy = -2*J

        dj = -2*self.j
        dw = self.x
        dj_dw = dj*dw
        
        dj_dc = dj*self.w
        
        self.w = self.w - self.lr*dj_dw
        return dj_dc, dj_dc
    
    def extract_output(self,x,exp):
        a = self.forward(x)
        self.j = self.calc_error(exp,a)
        ## Return error and outputs
        return [self.j,a]

    def calc_error(self,exp,a):
        if self.coss_fnc == "mse":
            return (exp - a)**2

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))