# Homework for Lecture 2

## Problem 1🌟:

Carefully study the backpropagation algorithm, on paper and in the program



# 🧠 Solution:

**Note:** all of the details are explained in the [2nd lecture](https://www.video.uni-erlangen.de/clip/id/11034) of the course, some figures are borrowed from [the slides](https://pad.gwdg.de/s/Machine_Learning_For_Physicists_2021#Slides)

Artificial Neural Networks (ANNs) consist of neurons and the connections between them. We've learned that the value of output neurons is calculated by applying a non-linear function <img src="https://render.githubusercontent.com/render/math?math=f(z)"> (e.g. Sigmoid, reLU) to a linear function of input neurons. In matrix/vector notation for one sample we have:
$$
\begin{split}
\text{input vector}&:\quad y^{in} \\
\text{weights matrix}&:\quad w \\
\text{bias vector}&:\quad b \\
\text{non-linear function}&: \quad f(z) \\
\text{output vector}&:\quad y \\ \\
y=f(z) \quad &, \quad z=wy^{in}
\end{split}
$$
Notice that the non-linear function <img src="https://render.githubusercontent.com/render/math?math=f(z)"> is kept fixed but the linear function *z* has parameters that can change. Basically we can say that "a neural network computes a complicated non-linear function that depends on all parameters (all weights and biases)". So we have:
$$
\begin{split}
y^{out}=F_w(y^{in}) \\
F: \text{neural network} \\
w: \text{weights and biases}
\end{split}
$$
Lets call the desired target function <img src="https://render.githubusercontent.com/render/math?math=F(y^{in})">. So we would like our neural network to approximate this target function:
$$
\begin{split}
y^{out} = F_w(y^{in}) \approx F(y^{in})
\end{split}
$$
We define a **Cost Function**:
$$
\begin{split}
C(w)=\frac{1}{2}\left<\left|\left| \underbrace{F_w(y^{in})-F(y^{in})}_{\text{Deviation}}
 \right|\right|^2\right>
\end{split}
$$
and <img src="https://render.githubusercontent.com/render/math?math=||.||"> denotes the vector norm. Because of taking average over all samples the *cost function* does not depend on the input sample, but it depends on the parameters of the neural network (namely weights and biases).



## ⛳️ Goal:

Our goal is to minimize the cost function by finding an appropriate set of weights and biases. For N samples we can write <img src="https://render.githubusercontent.com/render/math?math=C(w)"> as:
$$
\begin{split}
C(w)\approx \frac{1}{2}\frac{1}{N}\sum_{s=1}^{N} \left|\left|F_w(y^{(s)}) - F(y^{(s)}) \right|\right|^2 \quad,\quad s:\text{index of sample}
\end{split}
$$
Notice that for evaluating <img src="https://render.githubusercontent.com/render/math?math=C"> averaging over all training samples is a problem. The solution is to average over a few sample (a *batch* of samples). This way we get an approximated version of <img src="https://render.githubusercontent.com/render/math?math=C"> which we call it <img src="https://render.githubusercontent.com/render/math?math=\tilde C">:
$$
w_j \mapsto w_j-\eta\frac{\partial \tilde C(w)}{\partial w_j} \quad,\quad \eta: \,\text{step size (learning rate)}
$$
So the question is:

## 🤔 How to calculate <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial C}{\partial w_*}">?

(<img src="https://render.githubusercontent.com/render/math?math=w_*">: some weight or bias in the network) 

**Solution:** ⛓ *Chain rule* is here to help us.

---

### A Simple Example:

<img src="./imgs/p1_1.png" alt="simple example" />

---

For a full network we have to be careful with the notations and indices:
$$
\begin{split}
y_j^{(n)} &: \quad \text{value of neuron j in layer n}\\
z_j^{(n)} &: \quad \text{input value for }y=f(z) \\
w_{jk}^{(n,n-1)} &: \quad  \text{weight of connection between neuron k}  \\
&\quad \quad \text{in layer n-1 and neuron j in layer n}
\end{split}
$$
We have:
$$
C(w)=\left< C(w,y^{in}) \right>
$$
with <img src="https://render.githubusercontent.com/render/math?math=C(w,y^{in})"> being cost value for one particular input. If we assume that <img src="https://render.githubusercontent.com/render/math?math=y_{j}^{(n)}=f(z_{j}^{(n)})">  we can write:
$$
\begin{split}
\frac{\partial C(w,y^{in})}{\partial w_*} &= \sum_{j}\left(y_{j}^{(n)}-F_{j}\left(y^{\mathrm{in}}\right)\right) \frac{\partial y_{j}^{(n)}}{\partial w_{*}} \\
&= \sum_{j}\left(y_{j}^{(n)}-F_{j}\left(y^{\mathrm{in}}\right)\right) f^{\prime}(z_{j}^{(n)}) \frac{\partial z_{j}^{(n)}}{\partial w_{*}}
\end{split}
$$
Therefore we need to calculate <img src="https://render.githubusercontent.com/render/math?math=\partial z_{j}^{(n)}/\partial w_{*}">
$$
\begin{aligned} \frac{\partial z_{j}^{(n)}}{\partial w_{*}} &=\sum_{k} \frac{\partial z_{j}^{(n)}}{\partial y_{k}^{(n-1)}} \frac{\partial y_{k}^{(n-1)}}{\partial w_{*}} \\ &=\sum_{k} w_{j k}^{n, n-1} f^{\prime}(z_{k}^{(n-1)}) \frac{\partial z_{k}^{(n-1)}}{\partial w_{*}} 
\end{aligned}
$$
Therefore we need to calculate <img src="https://render.githubusercontent.com/render/math?math=\partial z_{k}^{(n-1)}/\partial w_{*}">
$$
\begin{aligned} \frac{\partial z_{k}^{(n-1)}}{\partial w_{*}} &=\sum_{l} \frac{\partial z_{k}^{(n-1)}}{\partial y_{l}^{(n-2)}} \frac{\partial y_{l}^{(n-2)}}{\partial w_{*}} \\ &=\sum_{l} w_{k l}^{n-1, n-2} f^{\prime}(z_{l}^{(n-2)}) \frac{\partial z_{l}^{(n-2)}}{\partial w_{*}} 
\end{aligned}
$$
We can see the pattern. Remember that we had:
$$
\frac{\partial z_{j}^{(n)}}{\partial w_{*}}=\sum_{k} w_{j k}^{n, n-1} f^{\prime}(z_{k}^{(n-1)}) \frac{\partial z_{k}^{(n-1)}}{\partial w_{*}}
$$
We can interpret <img src="https://render.githubusercontent.com/render/math?math=w_{j k}^{n,n-1} f^{\prime}\left(z_{k}^{(n-1)}\right)"> as a matrix multiplication with elements:
$$
M_{j k}^{(n, n-1)}=w_{j k}^{(n, n-1)} f^{\prime}(z_{k}^{(n-1)})
$$
By by repeating matrix multiplication and going down the network we have:
$$
\frac{\partial z_{j}^{(n)}}{\partial w_{*}}=\sum_{k, l, \ldots, u, v} M_{j k}^{n, n-1} M_{k l}^{n-1, n-2} \ldots M_{u v}^{\tilde{n}+1, \tilde{n}} \frac{\partial z_{v}^{(\tilde{n})}}{\partial w_{*}}
$$
So by going down the network we finally encounter the weight (or bias) which we wanted to calculate the derivative of the cost function. There are two cases:

1. if <img src="https://render.githubusercontent.com/render/math?math=w_{*}"> was really a weight:
   $$
   \frac{\partial z_j^{(n)}}{\partial w_{j,k}^{(\tilde n, \tilde n-1)}} = y_k^{(\tilde n -1)}
   $$
   
2. if <img src="https://render.githubusercontent.com/render/math?math=w_{*}"> was a bias:

$$
\frac{\partial z_j^{(n)}}{\partial b_{j}^{\tilde n}} = 1
$$

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
$$
\text{dWeights[n]}=\frac{\partial z_j^{(n)}}{\partial w_{jk}^{(n,n-1)}} = \left< \Delta_j y_k^{(n-1)} \right>
$$
and ```dBiases``` for:
$$
\text{dBiases[n]}=\frac{\partial z_j^{(n)}}{\partial b_{j}^{(n)}} = \left< \Delta_j  \right>
$$
So we implement these like this:

```python
dWeights[layer] = dot(transpose(y[lowe layer]) , Delta)/batchsize 
dBiases[layer]  = Delta.sum(0)/batchsize # summation over index 0 = batch index
```

And  to implement <img src="https://render.githubusercontent.com/render/math?math=\Delta_k">:
$$
\Delta_k = \sum_j \Delta_j M_{jk}^{(n,n-1)} \quad,\qquad M_{jk}^{(n,n-1)} = w_{jk}^{(n,n-1)} f^{\prime} (z_k^{(n-1)})
$$
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
