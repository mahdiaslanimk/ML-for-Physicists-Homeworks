# Homework for Lecture 2

## Problem 1🌟:

Carefully study the backpropagation algorithm, on paper and in the program



# 🧠 Solution:

**Note:** all of the details are explained in the [2nd lecture](https://www.video.uni-erlangen.de/clip/id/11034) of the course, some figures are borrowed from [the slides](https://pad.gwdg.de/s/Machine_Learning_For_Physicists_2021#Slides)

Artificial Neural Networks (ANNs) consist of neurons and the connections between them. We've learned that the value of output neurons is calculated by applying a non-linear function <img src="https://render.githubusercontent.com/render/math?math=f(z)"> (e.g. Sigmoid, reLU) to a linear function of input neurons. In matrix/vector notation for one sample we have:

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bsplit%7D%0A%5Ctext%7Binput%20vector%7D%26%3A%5Cquad%20y%5E%7Bin%7D%20%5C%5C%0A%5Ctext%7Bweights%20matrix%7D%26%3A%5Cquad%20w%20%5C%5C%0A%5Ctext%7Bbias%20vector%7D%26%3A%5Cquad%20b%20%5C%5C%0A%5Ctext%7Bnon-linear%20function%7D%26%3A%20%5Cquad%20f(z)%20%5C%5C%0A%5Ctext%7Boutput%20vector%7D%26%3A%5Cquad%20y%20%5C%5C%20%5C%5C%0Ay%3Df(z)%20%5Cquad%20%26%2C%20%5Cquad%20z%3Dwy%5E%7Bin%7D%0A%5Cend%7Bsplit%7D">

Notice that the non-linear function <img src="https://render.githubusercontent.com/render/math?math=f(z)"> is kept fixed but the linear function *z* has parameters that can change. Basically we can say that "a neural network computes a complicated non-linear function that depends on all parameters (all weights and biases)". So we have:

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bsplit%7D%0Ay%5E%7Bout%7D%3DF_w(y%5E%7Bin%7D)%20%5C%5C%0AF%3A%20%5Ctext%7Bneural%20network%7D%20%5C%5C%0Aw%3A%20%5Ctext%7Bweights%20and%20biases%7D%0A%5Cend%7Bsplit%7D">

Lets call the desired target function <img src="https://render.githubusercontent.com/render/math?math=F(y^{in})">. So we would like our neural network to approximate this target function:

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bsplit%7D%0Ay%5E%7Bout%7D%20%3D%20F_w(y%5E%7Bin%7D)%20%5Capprox%20F(y%5E%7Bin%7D)%0A%5Cend%7Bsplit%7D">

We define a **Cost Function**:

<img src="https://render.githubusercontent.com/render/math?math=C(w)%3D%5Cfrac%7B1%7D%7B2%7D%5Cleft%5Clangle%5Cleft%7C%5Cleft%7C%20%5Cunderbrace%7BF_w(y%5E%7Bin%7D)-F(y%5E%7Bin%7D)%7D_%7B%5Ctext%7BDeviation%7D%7D%5Cright%7C%5Cright%7C%5E2%20%5Cright%5Crangle">

and <img src="https://render.githubusercontent.com/render/math?math=||.||"> denotes the vector norm. Because of taking average over all samples the *cost function* does not depend on the input sample, but it depends on the parameters of the neural network (namely weights and biases).



## ⛳️ Goal:

Our goal is to minimize the cost function by finding an appropriate set of weights and biases. For N samples we can write <img src="https://render.githubusercontent.com/render/math?math=C(w)"> as:

<img src="https://render.githubusercontent.com/render/math?math=C(w)%5Capprox%20%5Cfrac%7B1%7D%7B2%7D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bs%3D1%7D%5E%7BN%7D%20%5Cleft%7C%5Cleft%7CF_w(y%5E%7B(s)%7D)%20-%20F(y%5E%7B(s)%7D)%20%5Cright%7C%5Cright%7C%5E2%20%5Cquad%2C%5Cquad%20s%3A%5Ctext%7Bindex%20of%20sample%7D">

Notice that for evaluating <img src="https://render.githubusercontent.com/render/math?math=C"> averaging over all training samples is a problem. The solution is to average over a few sample (a *batch* of samples). This way we get an approximated version of <img src="https://render.githubusercontent.com/render/math?math=C"> which we call it <img src="https://render.githubusercontent.com/render/math?math=\tilde C">:

<img src="https://render.githubusercontent.com/render/math?math=w_j%20%5Cmapsto%20w_j-%5Ceta%5Cfrac%7B%5Cpartial%20%5Ctilde%20C(w)%7D%7B%5Cpartial%20w_j%7D%20%5Cquad%2C%5Cquad%20%5Ceta%3A%20%5C%2C%5Ctext%7Bstep%20size%20(learning%20rate)%7D">

So the question is:

## 🤔 How to calculate <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial C}{\partial w_*}">?

(<img src="https://render.githubusercontent.com/render/math?math=w_*">: some weight or bias in the network) 

**Solution:** ⛓ *Chain rule* is here to help us.

---

### A Simple Example:

<img src="./imgs/p1_1.png" alt="simple example" />

---

For a full network we have to be careful with the notations and indices:

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bsplit%7D%0Ay_j%5E%7B(n)%7D%20%26%3A%20%5Cquad%20%5Ctext%7Bvalue%20of%20neuron%20j%20in%20layer%20n%7D%5C%5C%0Az_j%5E%7B(n)%7D%20%26%3A%20%5Cquad%20%5Ctext%7Binput%20value%20for%20%7Dy%3Df(z)%20%5C%5C%0Aw_%7Bjk%7D%5E%7B(n%2Cn-1)%7D%20%26%3A%20%5Cquad%20%20%5Ctext%7Bweight%20of%20connection%20between%20neuron%20k%7D%20%20%5C%5C%0A%26%5Cquad%20%5Cquad%20%5Ctext%7Bin%20layer%20n-1%20and%20neuron%20j%20in%20layer%20n%7D%0A%5Cend%7Bsplit%7D">

We have:

<img src="https://render.githubusercontent.com/render/math?math=C(w)%3D%5Cleft%5Clangle%20C(w%2Cy%5E%7Bin%7D)%20%5Cright%5Crangle">

with <img src="https://render.githubusercontent.com/render/math?math=C(w,y^{in})"> being cost value for one particular input. If we assume that <img src="https://render.githubusercontent.com/render/math?math=y_{j}^{(n)}=f(z_{j}^{(n)})">  we can write:

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bsplit%7D%0A%5Cfrac%7B%5Cpartial%20C(w%2Cy%5E%7Bin%7D)%7D%7B%5Cpartial%20w_*%7D%20%26%3D%20%5Csum_%7Bj%7D%5Cleft(y_%7Bj%7D%5E%7B(n)%7D-F_%7Bj%7D%5Cleft(y%5E%7B%5Cmathrm%7Bin%7D%7D%5Cright)%5Cright)%20%5Cfrac%7B%5Cpartial%20y_%7Bj%7D%5E%7B(n)%7D%7D%7B%5Cpartial%20w_%7B*%7D%7D%20%5C%5C%0A%26%3D%20%5Csum_%7Bj%7D%5Cleft(y_%7Bj%7D%5E%7B(n)%7D-F_%7Bj%7D%5Cleft(y%5E%7B%5Cmathrm%7Bin%7D%7D%5Cright)%5Cright)%20f%5E%7B%5Cprime%7D(z_%7Bj%7D%5E%7B(n)%7D)%20%5Cfrac%7B%5Cpartial%20z_%7Bj%7D%5E%7B(n)%7D%7D%7B%5Cpartial%20w_%7B*%7D%7D%0A%5Cend%7Bsplit%7D">

Therefore we need to calculate <img src="https://render.githubusercontent.com/render/math?math=\partial z_{j}^{(n)}/\partial w_{*}">

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Baligned%7D%20%5Cfrac%7B%5Cpartial%20z_%7Bj%7D%5E%7B(n)%7D%7D%7B%5Cpartial%20w_%7B*%7D%7D%20%26%3D%5Csum_%7Bk%7D%20%5Cfrac%7B%5Cpartial%20z_%7Bj%7D%5E%7B(n)%7D%7D%7B%5Cpartial%20y_%7Bk%7D%5E%7B(n-1)%7D%7D%20%5Cfrac%7B%5Cpartial%20y_%7Bk%7D%5E%7B(n-1)%7D%7D%7B%5Cpartial%20w_%7B*%7D%7D%20%5C%5C%20%26%3D%5Csum_%7Bk%7D%20w_%7Bj%20k%7D%5E%7Bn%2C%20n-1%7D%20f%5E%7B%5Cprime%7D(z_%7Bk%7D%5E%7B(n-1)%7D)%20%5Cfrac%7B%5Cpartial%20z_%7Bk%7D%5E%7B(n-1)%7D%7D%7B%5Cpartial%20w_%7B*%7D%7D%20%0A%5Cend%7Baligned%7D">

Therefore we need to calculate <img src="https://render.githubusercontent.com/render/math?math=\partial z_{k}^{(n-1)}/\partial w_{*}">

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Baligned%7D%20%5Cfrac%7B%5Cpartial%20z_%7Bk%7D%5E%7B(n-1)%7D%7D%7B%5Cpartial%20w_%7B*%7D%7D%20%26%3D%5Csum_%7Bl%7D%20%5Cfrac%7B%5Cpartial%20z_%7Bk%7D%5E%7B(n-1)%7D%7D%7B%5Cpartial%20y_%7Bl%7D%5E%7B(n-2)%7D%7D%20%5Cfrac%7B%5Cpartial%20y_%7Bl%7D%5E%7B(n-2)%7D%7D%7B%5Cpartial%20w_%7B*%7D%7D%20%5C%5C%20%26%3D%5Csum_%7Bl%7D%20w_%7Bk%20l%7D%5E%7Bn-1%2C%20n-2%7D%20f%5E%7B%5Cprime%7D(z_%7Bl%7D%5E%7B(n-2)%7D)%20%5Cfrac%7B%5Cpartial%20z_%7Bl%7D%5E%7B(n-2)%7D%7D%7B%5Cpartial%20w_%7B*%7D%7D%20%0A%5Cend%7Baligned%7D">

We can see the pattern. Remember that we had:

<img src="https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20z_%7Bj%7D%5E%7B(n)%7D%7D%7B%5Cpartial%20w_%7B*%7D%7D%3D%5Csum_%7Bk%7D%20w_%7Bj%20k%7D%5E%7Bn%2C%20n-1%7D%20f%5E%7B%5Cprime%7D(z_%7Bk%7D%5E%7B(n-1)%7D)%20%5Cfrac%7B%5Cpartial%20z_%7Bk%7D%5E%7B(n-1)%7D%7D%7B%5Cpartial%20w_%7B*%7D%7D">

We can interpret <img src="https://render.githubusercontent.com/render/math?math=w_{j k}^{n,n-1} f^{\prime}\left(z_{k}^{(n-1)}\right)"> as a matrix multiplication with elements:

<img src="https://render.githubusercontent.com/render/math?math=M_%7Bj%20k%7D%5E%7B(n%2C%20n-1)%7D%3Dw_%7Bj%20k%7D%5E%7B(n%2C%20n-1)%7D%20f%5E%7B%5Cprime%7D(z_%7Bk%7D%5E%7B(n-1)%7D)">

By by repeating matrix multiplication and going down the network we have:

<img src="https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20z_%7Bj%7D%5E%7B(n)%7D%7D%7B%5Cpartial%20w_%7B*%7D%7D%3D%5Csum_%7Bk%2C%20l%2C%20%5Cldots%2C%20u%2C%20v%7D%20M_%7Bj%20k%7D%5E%7Bn%2C%20n-1%7D%20M_%7Bk%20l%7D%5E%7Bn-1%2C%20n-2%7D%20%5Cldots%20M_%7Bu%20v%7D%5E%7B%5Ctilde%7Bn%7D%2B1%2C%20%5Ctilde%7Bn%7D%7D%20%5Cfrac%7B%5Cpartial%20z_%7Bv%7D%5E%7B(%5Ctilde%7Bn%7D)%7D%7D%7B%5Cpartial%20w_%7B*%7D%7D">

So by going down the network we finally encounter the weight (or bias) which we wanted to calculate the derivative of the cost function. There are two cases:

1. if <img src="https://render.githubusercontent.com/render/math?math=w_{*}"> was really a weight:
   
   <img src="https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20z_j%5E%7B(n)%7D%7D%7B%5Cpartial%20w_%7Bj%2Ck%7D%5E%7B(%5Ctilde%20n%2C%20%5Ctilde%20n-1)%7D%7D%20%3D%20y_k%5E%7B(%5Ctilde%20n%20-1)%7D">
   
2. if <img src="https://render.githubusercontent.com/render/math?math=w_{*}"> was a bias:

<img src="https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20z_j%5E%7B(n)%7D%7D%7B%5Cpartial%20b_%7Bj%7D%5E%7B%5Ctilde%20n%7D%7D%20%3D%201">

## Backpropagation Summary:

<img src="./imgs/p1_2.png" alt="BP Summary" />

Say we have 1M samples. We pick a set of samples or a mini-batch, say 100 samples. Out of this 100 samples we pick 1 training sample and we get 1 contribution in calculating the gradient (by doing steps 1 to 3). We do the same for all other samples in our mini-batch. We then take the average over the 100 contributions and finally take 1 step in updating the weights(biases). After taking one step, choose another mini-batch and do the same for it. We repeat this process until we arrive at the desired (minimized) cost function. **Keep in mind** that because we are batch processing we have to take care of an extra index in order to keep track of samples in the batch.

<img src="./imgs/p1_3.png" alt="BP Summary" />

## 🐍 Implementation

We are doing batch processing of many samples:

```python
y[layer]			 # Dimension: batchsize X neurons[layer]
Delta					 # Dimension: batchsize X neurons[layer]
Weights[layer] # Dimension: neurons[lower layer] X neurons[layer]
Biases[layer]  # Dimension: neurons[layer]
```

So use the names ```dWeights``` for:

\text{dWeights[n]}=\frac{\partial z_j^{(n)}}{\partial w_{jk}^{(n,n-1)}} = \left\langle \Delta_j y_k^{(n-1)} \right\rangle

and ```dBiases``` for:

<img src="https://render.githubusercontent.com/render/math?math=%5Ctext%7BdBiases%5Bn%5D%7D%3D%5Cfrac%7B%5Cpartial%20z_j%5E%7B(n)%7D%7D%7B%5Cpartial%20b_%7Bj%7D%5E%7B(n)%7D%7D%20%3D%20%5Cleft%5Clangle%20%5CDelta_j%20%20%5Cright%5Crangle">

So we implement these like this:

```python
dWeights[layer] = dot(transpose(y[lowe layer]) , Delta)/batchsize 
dBiases[layer]  = Delta.sum(0)/batchsize # summation over index 0 = batch index
```

And  to implement <img src="https://render.githubusercontent.com/render/math?math=\Delta_k">:

<img src="https://render.githubusercontent.com/render/math?math=%5CDelta_k%20%3D%20%5Csum_j%20%5CDelta_j%20M_%7Bjk%7D%5E%7B(n%2Cn-1)%7D%20%5Cquad%2C%5Cqquad%20M_%7Bjk%7D%5E%7B(n%2Cn-1)%7D%20%3D%20w_%7Bjk%7D%5E%7B(n%2Cn-1)%7D%20f%5E%7B%5Cprime%7D%20(z_k%5E%7B(n-1)%7D)">

we write:

```python
Delta = dot(Delta , transpose(Weights))*df_layer[lower layer] # Dimension: batchsize X neurons[lower layer]
```

here is an example for a NN with 3 layers (not counting the input layer):

<img src="./imgs/p1_4.png" alt="BP Summary" />

So we can implement the whole code (forward-pass and backward-pass) in about 30 lines of code:

```python
def net_f_df(z): # calculate f(z) and f'(z)
    val=1/(1+exp(-z)) # sigmoid
    return(val,exp(-z)*(val**2)) # return both f and f'
```

```python
def forward_step(y,w,b): # calculate values in next layer, from input y
    z=dot(y,w)+b # w=weights, b=bias vector for next layer
    return(net_f_df(z)) # apply nonlinearity and return result
```

```python
def apply_net(y_in): # one forward pass through the network
    global Weights, Biases, NumLayers
    global y_layer, df_layer # for storing y-values and df/dz values
    
    y=y_in # start with input values
    y_layer[0]=y
    for j in range(NumLayers): # loop through all layers
        # j=0 corresponds to the first layer above the input
        y,df=forward_step(y,Weights[j],Biases[j]) # one step
        df_layer[j]=df # store f'(z) [needed later in backprop]
        y_layer[j+1]=y # store f(z) [also needed in backprop]        
    return(y)
```

```python
def apply_net_simple(y_in): # one forward pass through the network
    # no storage for backprop (this is used for simple tests)

    y=y_in # start with input values
    y_layer[0]=y
    for j in range(NumLayers): # loop through all layers
        # j=0 corresponds to the first layer above the input
        y,df=forward_step(y,Weights[j],Biases[j]) # one step
    return(y)
```

```python
def backward_step(delta,w,df): 
    # delta at layer N, of batchsize x layersize(N))
    # w between N-1 and N [layersize(N-1) x layersize(N) matrix]
    # df = df/dz at layer N-1, of batchsize x layersize(N-1)
    return( dot(delta,transpose(w))*df )
```

```python
def backprop(y_target): # one backward pass through the network
    # the result will be the 'dw_layer' matrices that contain
    # the derivatives of the cost function with respect to
    # the corresponding weight
    global y_layer, df_layer, Weights, Biases, NumLayers
    global dw_layer, db_layer # dCost/dw and dCost/db (w,b=weights,biases)
    global batchsize
    
    delta=(y_layer[-1]-y_target)*df_layer[-1]
    dw_layer[-1]=dot(transpose(y_layer[-2]),delta)/batchsize
    db_layer[-1]=delta.sum(0)/batchsize
    for j in range(NumLayers-1):
        delta=backward_step(delta,Weights[-1-j],df_layer[-2-j])
        dw_layer[-2-j]=dot(transpose(y_layer[-3-j]),delta)
        db_layer[-2-j]=delta.sum(0)/batchsize
```

## Training the Network:

```python
def gradient_step(eta): # update weights & biases (after backprop!)
    global dw_layer, db_layer, Weights, Biases
    
    for j in range(NumLayers):
        Weights[j]-=eta*dw_layer[j]
        Biases[j]-=eta*db_layer[j]
```

```python
def train_net(y_in,y_target,eta): # one full training batch
    # y_in is an array of size batchsize x (input-layer-size)
    # y_target is an array of size batchsize x (output-layer-size)
    # eta is the stepsize for the gradient descent
    global y_out_result
    
    y_out_result=apply_net(y_in)
    backprop(y_target)
    gradient_step(eta)
    cost=((y_target-y_out_result)**2).sum()/batchsize
    return(cost)
```



