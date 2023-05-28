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
        print("The Mosttt " , len(self.layers))

    def param_check(self):
        ## Check the parameters
        assert type(self.layer_width) == list # Should be array

    def calc_output(self,x,ref):
        exp = self.forward(x)
        j = self.layers[-1].extract_output(exp,ref)

        return [j,exp,x]
    
    
    def forward(self,x):
        temp = x
        
        for i in range (0,self.hidden_size):
            print("\n \n |||| In neuron {}".format(i))
            a = self.layers[i].forward(temp)
            y = self.layers[i].non_linear("sig",a)
            temp = y

        print("\n \n |||| In neuron {}".format(self.hidden_size))
        a = self.layers[-1].forward(temp)

        return a
    
    def back_prop(self):
        cnt = self.hidden_size -1 
        da_dc = None
        for a in range (0,self.hidden_size+1):
            cnt = self.hidden_size-a
            da_dc = self.layers[cnt].backprop(da_dc)
            print(" Backproped @ Neuron: {} ".format(cnt))
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
        self.w = np.random.rand(layer_width,input_size)
        self.b = np.random.rand(layer_width,1)

        ########################################
        ###### STORED VALUES FOR BACKPROP !! ###
        ########################################
        ## Store sigmoid 
        self.sig = np.zeros([layer_width,1])
        ## Store x
        self.x = np.zeros([input_size,1])

    def forward(self,x):       
        print("Forward Calculations || W.x + b || == a \n")
        print("W -> \n {}, x -> \n {}, b -> \n {}\n----".format(self.w,x,self.b))
        ##  a = w*x + b
        self.param_check(x = x)
        self.x = x
        a = np.dot(self.w,x)
        a = a + self.b
        print("a -> {}".format(a))
        return a
    
    def non_linear(self,fnc,a):
        ## σ(a) = y
        self.param_check(fnc = fnc,a = a)

        if fnc == "sig":
            sig = 1.0/(1.0 + np.exp(-a))
        self.sig = sig
        print("Nonlinear ||σ(a) = y|| == a \n")
        print("a -> {} ".format(a))
        print("σ(a) -> {}".format(self.sig))
        return sig

    def param_check(self,**kwargs):

        for args in kwargs:
            if args == "fnc":
                assert type(kwargs[args]) == str
            if args == "a":
                assert np.shape(kwargs[args]) == (self.layer_width,1)
            if args == "x":
                assert type(kwargs[args]) == np.ndarray
                assert np.shape(kwargs[args]) == np.shape(self.x)
            if args == "sig":
                assert np.shape(kwargs[args]) == (self.layer_width,1)
     
    def clear(self):
        self.sig = np.zeros([self.layer_width,1])
        self.x = np.zeros([self.input_size,1])

    def backprop(self,dJ_da):
        ####
        ####  w*c_t + b = a_(t+1)
        ####  c_(t+1) = σ(a_(t+1))
        ####
        ####  da_dc(t)= w
        ####  dc(t+1)_da = -c(1-c)

        self.param_check(sig = self.sig)

        dJ_da = dJ_da * self.sig.T

        dJ_dw = dJ_da.T @ self.x.T

        da_dc = self.w

        dJ_dc = dJ_da @ da_dc

        dJ_da = dJ_dc

        #dJ_dw = np.clip(dJ_dw,-50,50)
        self.w = self.w + self.lr*dJ_dw

        return dJ_da
    
class output_layer(neuron_layer):
    def __init__(self,input_size,output_num,coss_fnc="mse",learning_rate=0.0005):
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

        dJ_da = -2*self.j
        dJ_da = dJ_da.T

        dJ_dw = dJ_da.T @ self.x.T

        da_dc = self.w
        
        dJ_dc = dJ_da @ da_dc
        dJ_da = dJ_dc

        self.w = self.w + self.lr*dJ_dw

        return dJ_da
    
    def extract_output(self,exp,ref):
        
        self.j = self.calc_error(exp,ref)
        ## Return error and outputs
        return self.j

    def calc_error(self,exp,ref):
        if self.coss_fnc == "mse":
            return (ref - exp)**2

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))