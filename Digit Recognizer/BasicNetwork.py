
from time import time
import numpy as np
class BasicNetwork3:
    def __init__(self,level_details,use_random_weight=True,input_normalize_to=1,activation_scale_to=1,learning_curve_rate=1):
        self.act_levels=level_details
        self.num_layers=len(self.act_levels)
        self.weights=[]
        self.biases=[]
        self.use_random_weights=use_random_weight
        self.normalize_to=input_normalize_to
        self.number_of_classes=self.act_levels[-1]
        self.activation_scale_to=activation_scale_to
        self.learning_rate=learning_curve_rate
        self.gen_random_w_b()
        
    def normalize(self,x):
        return ((x-min(x))/(max(x)-min(x)))*self.normalize_to
        
    def gen_random_w_b(self):
        self.biases = [np.random.randn(y, 1).reshape(y) for y in self.act_levels[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.act_levels[:-1], self.act_levels[1:])]
        
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    
    def feed_forward(self,a):
        a=np.array(self.normalize(a))
        for w,b in zip(self.weights,self.biases):
            a=sigmoid(np.dot(w,a)+b)
        return a
    
    def gen_y(self,x):
        required=[]
        for i in range(self.number_of_classes):
            if i==x:
                required.append(1.0)
            else:
                required.append(0.0)
        return np.array(required)
    
    def backprop(self,x,y):
        nabla_b = np.array([np.zeros(b.shape) for b in self.biases])
        nabla_w = np.array([np.zeros(w.shape) for w in self.weights])
        # feedforward
        activation = np.array(self.normalize(x))
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = np.array(self.cost_derivative(activations[-1], self.gen_y(y)) * sigmoid_prime(zs[-1]))
        nabla_b[-1] = delta
        nabla_w[-1] = np.outer(delta, activations[-2]) 
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.array(np.dot(self.weights[-l+1].transpose(), delta) * sp)
            nabla_b[-l] = delta
            nabla_w[-l] = np.outer(delta, activations[-l-1])
        return (nabla_b, nabla_w)
        
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)    
    
    
    def update_mini_batch(self, mini_batch, eta):
        nabla_b=np.array([np.zeros(b.shape) for b in self.biases])
        nabla_w = np.array([np.zeros(w.shape) for w in self.weights])
        
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w=self.backprop(x,y)
            nabla_b+=delta_nabla_b
            nabla_w+=delta_nabla_w
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        
    def fit(self,training,labels,epochs=1,mini_batch_size=10):
        
        training_data=[]
        for i in range(len(training)):
            training_data.append(tuple([training[i],labels[i]]))
        
        training_data = list(training_data)
        n = len(training_data)

        test_data=training_data[int(0.1*len(training_data)):int(0.5*len(training_data))]
        n_test=len(test_data)
        
        for j in range(epochs):
            start_time=time()
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, self.learning_rate)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test));
            else:
                print("Epoch {} complete".format(j))
            print(time()-start_time)
        
    def predict(self,test):
        test_results=[np.argmax(self.feed_forward(x)) for x in test]
        return test_results
    
    
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def accuracy(labels,y1):
        count=0
        for i in range(len(y1)):
            if labels[i]==y1[i]:
                count+=1
        return count/len(labels)
    