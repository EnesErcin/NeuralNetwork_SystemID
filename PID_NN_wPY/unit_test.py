import numpy as np

### Parameters
num_of_input = 3
num_of_output = 3
hidden_size = [num_of_input,2,4,num_of_output]
layer_cnt = len(hidden_size)-1
###
"""
my_weigths = []
my_inputs =np.ones((hidden_size[0],1))


class my_linears():
    def __init__(self,layer_width,input_size,coef = 1):
        self.w = np.ones((hidden_size[i+1],hidden_size[i]))*coef
        self.x = np.zeros([input_size,1])
        self.layer_width = layer_width
        self.input_size = input_size
    
    def forward(self,x):
        ##  a = w*x + b
        self.x = x

        a = np.dot(self.w,x)
        return a

    
    def backprop(self,de_dw):
        dw = self.x
        print(de_dw)
        print("Shape \t" ,np.shape(de_dw))
        return [dw,de_dw]
    

layers = []
## Generate  >> input and  hidden layers

for i in range (0,layer_cnt):
    my_neuron = my_linears(hidden_size[i],hidden_size[i+1])
    layers.append(my_neuron)
    
    

print("------Feed Forward-------- \n \n")
for i in range (0, layer_cnt):
    my_inputs = layers[i].forward(my_inputs)
    print(my_inputs)
    print("Shape \t" , np.shape(my_inputs))


print("------Derivatives-------- \n \n")
de_dw = my_inputs
for i in range(0,layer_cnt):
    cnt = layer_cnt-i-1
    de_dw,_ = layers[cnt].backprop(de_dw)
    print("Hello",type(de_dw))

    print("\n \n")


"""

my_a = np.random.randn(1,3)
my_x2 =np.ones((3,4))
my_res = np.matmul(my_a,my_x2)
print(my_res)
print(np.shape(my_res))


print("\n\n")
print(my_a)
print(my_x2)