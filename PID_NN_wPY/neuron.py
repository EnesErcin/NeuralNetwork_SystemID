import numpy as np
from basic_fncts import sigmoid

def myloop(method):
    def inner(refs,x):
        ## Store every result into an array
        proc = []

        def fnc_single_nn(refs,x):
            return method(refs,x)
        
        for i in range (0,refs.hidden_size):
            ## Apply the method for every neuron
            proc.append(fnc_single_nn(refs.layers[i],x))
        
        return proc
    return inner



class neuron_layer:
    def __init__(self,input_size,layer_width):

        ####### |  | |  |   |  |    |  | |  |    |  |    |  |  |  |     |  | #####
        ####### |  | |  |   |  |    |  | |  |    |  |    |  |  |  |     |  | #####
        ####### |x1|*|w1| + |b1| ,  |x1|*|w2|  + |b2| ,  |x1| *|w3| +   |b3| #####
        ####### |  | |  |   |  |    |  | |  |    |  |    |  |  |  |     |  | #####

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
        self.clear()
        ##  a = w*x + b
        self.x = x

        a = np.dot(self.w,x)
        a = a + self.b
        return a
    
    def non_linear(self,fnc,a):
        ## σ(a) = y
        self.param_check(fnc,a)
        if fnc == "sig":
            for i in range (0,self.layer_width):
                self.sig[i] = (sigmoid(a[i]))
        return self.sig

    def clear(self):
        self.sig = np.zeros([self.layer_width,1])
        self.x = np.zeros([self.input_size,1])

    def param_check(self,fnc,a):
        assert type(fnc) == str
        assert np.shape(a) == (self.layer_width,1) 
            
class output_layer(neuron_layer):
    def __init__(self,input_size,output_num,coss_fnc):
        neuron_layer.__init__(self,input_size,output_num)
        self.coss_fnc = coss_fnc
        ## Store error
        self.j = np.zeros([self.layer_width,1])

    def forward(self,x):
        self.clear()
        ##  a = w*x + b
        self.x = x

        a = np.dot(self.w,x)
        a = a + self.b
        return a
    
    def extract_output(self,x,exp):
        a = self.forward(x)
        self.j = self.calc_error(exp,a)
        ## Return error and outputs
        return [self.j,a]

    def calc_error(self,exp,a):
        if self.coss_fnc == "mse":
            return (exp - a)**2


    def non_linear(self):
        # Output layer does not have activatison function in this case
        assert False 

    def clear(self):
        self.j = np.zeros([self.layer_width,1])
        self.x = np.zeros([self.input_size,1])

    def param_check(self,fnc,a):
        assert type(fnc) == str
        assert np.shape(a) == (self.layer_width,1) 




class neuron_layers:
    def __init__(self,hidden_layers):
        self.hidden_layers = hidden_layers
        self.param_check()
        self.hidden_size = len(hidden_layers) -1 # Include input at first || Output size at last
        

        self.layers = []
        ## Generate  >> input and  hidden layers
        for i in range (0,self.hidden_size-1):
            my_neuron = neuron_layer(hidden_layers[i],hidden_layers[i+1])
            self.layers.append(my_neuron)
        
        ## Generate >> output layer
        my_outlayer = output_layer(hidden_layers[-2],hidden_layers[-1],"mse")
        self.layers.append(my_outlayer)


    def forward(self,x):
        temp = []
        temp.append(x)

        for i in range (0,self.hidden_size-1):
            print("\n \n |||| In neuron {}".format(i))
            a = self.layers[i].forward(temp[i])
            y = self.layers[i].non_linear("sig",a)
            y = np.reshape(y,(np.size(y),1))
            temp.append(y)

        return temp

    def calc_output(self,x,exp):
        temp = self.forward(x)

        print("\n \n |||| In neuron {}".format(self.hidden_size))
        [j,a] = self.layers[-1].extract_output(temp[-1],exp)
        a = np.reshape(a,(np.size(a),1))
        temp.append(a)

        return [j,a]

    def param_check(self):
        ## Check the parameters
        assert type(self.hidden_layers) == list # Should be array