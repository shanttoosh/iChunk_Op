# Neural Networks

## Table of Contents
1. [Fundamentals](#fundamentals)
2. [Backpropagation](#backpropagation)
3. [Activation Functions](#activation-functions)
4. [Optimization Algorithms](#optimization-algorithms)
5. [Regularization](#regularization)
6. [Architecture Patterns](#architecture-patterns)
7. [Interview Insights](#interview-insights)

---

## Fundamentals

### What is a Neural Network?

**Definition**: Computational model inspired by biological neurons that learns to map inputs to outputs.

### Basic Architecture

```
Input Layer    Hidden Layers    Output Layer
    
x₁  ○─────┐
          ├─→ ○─────┐
x₂  ○─────┤         ├─→ ○─────→ ŷ₁
          ├─→ ○─────┤
x₃  ○─────┘         ├─→ ○─────→ ŷ₂
                    │
                 ○──┘
```

### Single Neuron (Perceptron)

```
Inputs: x₁, x₂, ..., xₙ
         ↓
Weights: w₁, w₂, ..., wₙ
         ↓
Weighted Sum: z = Σ(wᵢxᵢ) + b
         ↓
Activation: a = σ(z)
         ↓
Output: a
```

**Mathematical Formulation**:

$$z = \sum_{i=1}^n w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$

$$a = \sigma(z)$$

Where:
- $\mathbf{x}$ = input vector
- $\mathbf{w}$ = weight vector
- $b$ = bias term
- $\sigma$ = activation function

### Forward Pass

**Layer-by-layer computation**:

For layer $l$:

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

$$\mathbf{a}^{[l]} = \sigma^{[l]}(\mathbf{z}^{[l]})$$

Where:
- $\mathbf{a}^{[0]} = \mathbf{x}$ (input)
- $\mathbf{W}^{[l]}$ = weight matrix for layer $l$
- $\mathbf{b}^{[l]}$ = bias vector for layer $l$

**Pseudocode**:

```python
def forward_pass(X, weights, biases, activations):
    """
    X: [batch_size, input_dim]
    weights: List of weight matrices
    biases: List of bias vectors
    activations: List of activation functions
    """
    a = X
    layer_outputs = [a]
    
    for W, b, activation in zip(weights, biases, activations):
        # Linear transformation
        z = matmul(a, W) + b  # [batch_size, layer_dim]
        
        # Activation
        a = activation(z)
        
        layer_outputs.append(a)
    
    return a, layer_outputs  # Final output and all intermediate activations
```

### Universal Approximation Theorem

**Theorem**: A feedforward neural network with:
- Single hidden layer
- Sufficient neurons
- Non-linear activation

Can approximate any continuous function to arbitrary precision.

**Implication**: Neural networks are powerful function approximators!

**Limitation**: Doesn't say how many neurons needed or how to train.

---

## Backpropagation

### Core Concept

**Goal**: Compute gradients of loss with respect to all weights.

**Method**: Chain rule applied backwards through network.

### The Math

**Loss Function**: $\mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})$

**Goal**: Compute $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$ and $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}}$ for all layers.

### Chain Rule

For a function $f(g(x))$:

$$\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

### Backpropagation Algorithm

**Step 1**: Forward pass (compute all activations)

**Step 2**: Compute output layer gradient

$$\delta^{[L]} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[L]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \odot \sigma'^{[L]}(\mathbf{z}^{[L]})$$

Where $\odot$ is element-wise multiplication.

**Step 3**: Backpropagate error

For $l = L-1, L-2, ..., 1$:

$$\delta^{[l]} = (\mathbf{W}^{[l+1]})^T \delta^{[l+1]} \odot \sigma'^{[l]}(\mathbf{z}^{[l]})$$

**Step 4**: Compute weight gradients

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \delta^{[l]} (\mathbf{a}^{[l-1]})^T$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \delta^{[l]}$$

### Example: Two-Layer Network

```
Network:
x → [W¹, b¹, ReLU] → h → [W², b², Sigmoid] → ŷ

Loss: L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

**Forward**:
```python
z1 = W1 @ x + b1
h = relu(z1)
z2 = W2 @ h + b2
y_hat = sigmoid(z2)
loss = binary_cross_entropy(y, y_hat)
```

**Backward**:
```python
# Output layer
dL_dy_hat = -(y / y_hat - (1 - y) / (1 - y_hat))
dy_hat_dz2 = sigmoid_derivative(z2)  # y_hat * (1 - y_hat)
delta2 = dL_dy_hat * dy_hat_dz2

# Gradients for W2, b2
dL_dW2 = delta2 @ h.T
dL_db2 = delta2

# Hidden layer
delta1 = (W2.T @ delta2) * relu_derivative(z1)

# Gradients for W1, b1
dL_dW1 = delta1 @ x.T
dL_db1 = delta1
```

### Computational Graph View

```
       x
       ↓
   ┌───────┐
   │  W¹x  │ → z¹
   └───────┘
       ↓
   ┌───────┐
   │ ReLU  │ → h
   └───────┘
       ↓
   ┌───────┐
   │  W²h  │ → z²
   └───────┘
       ↓
   ┌───────┐
   │Sigmoid│ → ŷ
   └───────┘
       ↓
   ┌───────┐
   │  Loss │ → L
   └───────┘

Backward: Flow gradients in reverse
L → ∂L/∂ŷ → ∂L/∂z² → ∂L/∂h → ∂L/∂z¹ → ∂L/∂W¹
```

### Implementation

```python
class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: [input_dim, hidden1, hidden2, ..., output_dim]
        """
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X):
        """Forward pass"""
        self.layer_inputs = [X]
        self.layer_outputs = [X]
        
        a = X
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            z = a @ W + b
            self.layer_inputs.append(z)
            a = relu(z)
            self.layer_outputs.append(a)
        
        # Output layer (no activation for regression, sigmoid/softmax for classification)
        z = a @ self.weights[-1] + self.biases[-1]
        self.layer_inputs.append(z)
        a = sigmoid(z)
        self.layer_outputs.append(a)
        
        return a
    
    def backward(self, X, y, y_pred):
        """Backpropagation"""
        m = X.shape[0]  # Batch size
        gradients_w = []
        gradients_b = []
        
        # Output layer gradient
        dL_da = -(y / y_pred - (1 - y) / (1 - y_pred))
        da_dz = sigmoid_derivative(self.layer_inputs[-1])
        delta = dL_da * da_dz
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradients
            dW = (self.layer_outputs[i].T @ delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            gradients_w.insert(0, dW)
            gradients_b.insert(0, db)
            
            # Propagate delta to previous layer
            if i > 0:
                delta = (delta @ self.weights[i].T) * relu_derivative(self.layer_inputs[i])
        
        return gradients_w, gradients_b
    
    def train_step(self, X, y, learning_rate=0.01):
        """Single training step"""
        # Forward
        y_pred = self.forward(X)
        
        # Backward
        grad_w, grad_b = self.backward(X, y, y_pred)
        
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grad_w[i]
            self.biases[i] -= learning_rate * grad_b[i]
        
        # Compute loss
        loss = binary_cross_entropy(y, y_pred)
        return loss
```

### Vanishing/Exploding Gradients

**Problem**: Gradients become too small or too large as they backpropagate.

**Vanishing Gradient**:

For deep networks with sigmoid activation:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[1]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[L]}} \cdot \prod_{l=1}^{L-1} \frac{\partial \mathbf{z}^{[l+1]}}{\partial \mathbf{z}^{[l]}}$$

If $|\sigma'(z)| < 1$ (e.g., sigmoid: $\sigma'(z) \leq 0.25$), product approaches 0.

**Solutions**:
1. **ReLU activation**: Gradient is 1 for positive inputs
2. **Batch Normalization**: Normalizes activations
3. **Residual connections**: Skip connections allow gradient flow
4. **Careful initialization**: Xavier, He initialization

**Exploding Gradient**:

Gradients grow exponentially, causing unstable training.

**Solutions**:
1. **Gradient clipping**: Cap gradient magnitude
2. **Lower learning rate**: Smaller update steps
3. **Batch normalization**: Stabilizes activations

```python
def clip_gradients(gradients, max_norm=1.0):
    """Clip gradients by global norm"""
    total_norm = 0
    for grad in gradients:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for grad in gradients:
            grad *= clip_coef
    
    return gradients
```

---

## Activation Functions

### Why Non-linearity?

**Without activation**: Network collapses to linear transformation.

$$h = W_2(W_1 x) = (W_2 W_1) x = W_{combined} x$$

**With activation**: Can learn complex non-linear patterns.

### Common Activation Functions

#### 1. Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Derivative**:
$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Range**: (0, 1)

**Pros**: Smooth, interpretable as probability  
**Cons**: Vanishing gradient, not zero-centered

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
```

**Use**: Output layer for binary classification

#### 2. Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

**Derivative**:
$$\tanh'(x) = 1 - \tanh^2(x)$$

**Range**: (-1, 1)

**Pros**: Zero-centered, stronger gradients than sigmoid  
**Cons**: Still vanishing gradient

**Use**: RNN hidden states

#### 3. ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$$

**Derivative**:
$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$$

**Pros**: No vanishing gradient, computationally efficient, sparse activation  
**Cons**: "Dying ReLU" (neurons stuck at 0)

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
```

**Use**: Default choice for hidden layers

#### 4. Leaky ReLU

$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{otherwise} \end{cases}$$

Typically $\alpha = 0.01$

**Derivative**:
$$\text{LeakyReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{otherwise} \end{cases}$$

**Pros**: Fixes dying ReLU problem  
**Cons**: Extra hyperparameter

#### 5. GELU (Gaussian Error Linear Unit)

$$\text{GELU}(x) = x \cdot \Phi(x)$$

Where $\Phi(x)$ is cumulative distribution function of standard normal.

**Approximation**:
$$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right)$$

**Pros**: Smooth, used in BERT, GPT  
**Cons**: More expensive to compute

```python
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
```

**Use**: Transformer models

#### 6. Swish (SiLU)

$$\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

**Pros**: Self-gated, smooth, outperforms ReLU in some tasks  
**Cons**: More compute

**Use**: Modern architectures (EfficientNet)

### Comparison

| Function | Range | Gradient | Zero-Centered | Use Case |
|----------|-------|----------|---------------|----------|
| **Sigmoid** | (0, 1) | Vanishing | No | Binary output |
| **Tanh** | (-1, 1) | Vanishing | Yes | RNN |
| **ReLU** | [0, ∞) | Healthy | No | Default hidden |
| **Leaky ReLU** | (-∞, ∞) | Healthy | No | Fix dying ReLU |
| **GELU** | (-∞, ∞) | Healthy | No | Transformers |
| **Swish** | (-∞, ∞) | Healthy | No | Modern CNNs |

---

## Optimization Algorithms

### Gradient Descent Variants

#### 1. Batch Gradient Descent

**Update** using entire dataset:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

**Pros**: Stable convergence  
**Cons**: Slow for large datasets, can't escape local minima

#### 2. Stochastic Gradient Descent (SGD)

**Update** using single sample:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(x_i, y_i; \theta_t)$$

**Pros**: Fast updates, can escape local minima (noise)  
**Cons**: Noisy updates, unstable convergence

#### 3. Mini-Batch Gradient Descent

**Update** using small batch (typically 32-256):

$$\theta_{t+1} = \theta_t - \eta \frac{1}{B} \sum_{i=1}^B \nabla_\theta \mathcal{L}(x_i, y_i; \theta_t)$$

**Pros**: Balance between speed and stability  
**Cons**: Batch size is hyperparameter

### Advanced Optimizers

#### 1. SGD with Momentum

**Idea**: Accumulate gradient direction over time.

$$v_t = \beta v_{t-1} + (1 - \beta) \nabla_\theta \mathcal{L}(\theta_t)$$

$$\theta_{t+1} = \theta_t - \eta v_t$$

Typically $\beta = 0.9$

**Analogy**: Ball rolling down hill with momentum.

```python
class SGDMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        
        for i in range(len(params)):
            self.velocity[i] = self.momentum * self.velocity[i] + (1 - self.momentum) * grads[i]
            params[i] -= self.lr * self.velocity[i]
```

#### 2. RMSprop

**Idea**: Adapt learning rate for each parameter based on gradient history.

$$s_t = \beta s_{t-1} + (1 - \beta) (\nabla_\theta \mathcal{L})^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t + \epsilon}} \nabla_\theta \mathcal{L}$$

**Benefits**: 
- Large gradients → smaller updates
- Small gradients → larger updates

```python
class RMSprop:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.cache = None
    
    def update(self, params, grads):
        if self.cache is None:
            self.cache = [np.zeros_like(p) for p in params]
        
        for i in range(len(params)):
            self.cache[i] = self.beta * self.cache[i] + (1 - self.beta) * grads[i]**2
            params[i] -= self.lr * grads[i] / (np.sqrt(self.cache[i]) + self.epsilon)
```

#### 3. Adam (Adaptive Moment Estimation)

**Combines** momentum + RMSprop.

**Update Rules**:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta \mathcal{L}$$  (First moment, momentum)

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_\theta \mathcal{L})^2$$  (Second moment, RMSprop)

**Bias Correction**:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Parameter Update**:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Default values**: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 0.001$

```python
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        
        for i in range(len(params)):
            # Update biased first moment
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            
            # Update biased second moment
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i]**2
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Update parameters
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

#### 4. AdamW

**Enhancement**: Decouple weight decay from gradient-based update.

$$\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$

Where $\lambda$ is weight decay coefficient.

**Why better**: Weight decay applied correctly, improves generalization.

### Optimizer Comparison

| Optimizer | Pros | Cons | Use Case |
|-----------|------|------|----------|
| **SGD** | Simple, generalizes well | Slow convergence | With momentum when tuning carefully |
| **SGD+Momentum** | Faster convergence | Still needs LR tuning | Computer vision |
| **RMSprop** | Adaptive LR | Can be unstable | RNNs |
| **Adam** | Fast, robust | Can overfit | Default choice |
| **AdamW** | Best generalization | Slightly complex | Transformers, LLMs |

---

## Regularization

### Why Regularization?

**Problem**: Model memorizes training data (overfitting).

**Goal**: Improve generalization to unseen data.

### Techniques

#### 1. L2 Regularization (Weight Decay)

**Add penalty** for large weights:

$$\mathcal{L}_{total} = \mathcal{L}_{data} + \frac{\lambda}{2} \sum_{l} ||\mathbf{W}^{[l]}||^2$$

**Effect**: Encourages smaller weights, smoother function.

```python
def l2_loss(weights, lambda_reg):
    return lambda_reg / 2 * sum(np.sum(W**2) for W in weights)

# In training
loss = data_loss + l2_loss(model.weights, lambda_reg=0.01)
```

#### 2. L1 Regularization

$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \sum_{l} ||\mathbf{W}^{[l]}||_1$$

**Effect**: Encourages sparsity (many weights → 0).

#### 3. Dropout

**Idea**: Randomly "drop" neurons during training.

```python
def dropout(x, dropout_rate=0.5, training=True):
    """
    x: [batch, features]
    dropout_rate: Probability of dropping
    """
    if not training:
        return x  # No dropout during inference
    
    # Create mask
    mask = np.random.rand(*x.shape) > dropout_rate
    
    # Scale to maintain expected value
    return x * mask / (1 - dropout_rate)
```

**Benefits**:
- Prevents co-adaptation of neurons
- Ensemble effect (different subnetworks each iteration)

**Typical rates**: 0.2-0.5

#### 4. Batch Normalization

**Idea**: Normalize activations within each mini-batch.

For layer input $x$:

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y = \gamma \hat{x} + \beta$$

Where:
- $\mu_B, \sigma_B^2$ = batch mean, variance
- $\gamma, \beta$ = learnable scale and shift

```python
class BatchNorm:
    def __init__(self, dim, epsilon=1e-5, momentum=0.9):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.epsilon = epsilon
        self.momentum = momentum
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)
    
    def forward(self, x, training=True):
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        
        # Scale and shift
        out = self.gamma * x_norm + self.beta
        
        return out
```

**Benefits**:
- Reduces internal covariate shift
- Allows higher learning rates
- Acts as regularizer

#### 5. Early Stopping

**Idea**: Stop training when validation loss stops improving.

```python
def train_with_early_stopping(model, train_data, val_data, patience=10):
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # Train
        train_loss = train_epoch(model, train_data)
        
        # Validate
        val_loss = validate(model, val_data)
        
        # Check improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stop
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            load_checkpoint(model)  # Restore best model
            break
```

#### 6. Data Augmentation

**Idea**: Create variations of training data.

**Examples**:
- Images: Rotation, flip, crop, color jitter
- Text: Synonym replacement, back-translation
- Audio: Time stretch, pitch shift

---

## Architecture Patterns

### 1. Residual Connections (ResNet)

**Problem**: Deep networks hard to train (vanishing gradients).

**Solution**: Skip connections.

```
    x
    │
    ├─────────┐
    ↓         │
  Conv        │
    ↓         │
  ReLU        │
    ↓         │
  Conv        │
    ↓         │
    + ←───────┘
    ↓
  ReLU
    ↓
   out
```

**Formula**:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

**Benefits**: Enables training very deep networks (100+ layers).

### 2. Dense Connections (DenseNet)

**Each layer** connected to all previous layers.

```
x₀ → Layer 1 → x₁ ─┐
  └────────────────┼→ Layer 2 → x₂ ─┐
                   └────────────────┼→ Layer 3 → x₃
```

**Formula**:

$$\mathbf{x}_l = \mathcal{H}_l([\mathbf{x}_0, \mathbf{x}_1, ..., \mathbf{x}_{l-1}])$$

**Benefits**: Feature reuse, reduces parameters.

### 3. Attention Mechanisms

See [Transformers guide](01-LLM-AND-TRANSFORMERS.md) for details.

---

## Interview Insights

### Common Questions

**Q1: Derive backpropagation for a 2-layer network.**

**Answer**: [Provide step-by-step derivation shown in Backpropagation section]

**Q2: Why ReLU better than sigmoid?**

**Answer**:
1. **No vanishing gradient**: Gradient is 1 for x > 0
2. **Sparse activation**: ~50% neurons inactive
3. **Computationally efficient**: max(0, x) vs exponential
4. **Empirically better**: Converges faster

**Q3: Explain Adam optimizer.**

**Answer**: Combines momentum (first moment) and RMSprop (second moment). Uses bias correction for early iterations. Adaptive learning rate per parameter. Default choice due to robustness and fast convergence.

**Q4: How does batch normalization help?**

**Answer**:
1. Reduces internal covariate shift
2. Allows higher learning rates (stable gradients)
3. Reduces dependence on initialization
4. Acts as regularizer (batch statistics add noise)
5. Enables deeper networks

### Common Pitfalls

❌ **Forgetting bias correction in Adam**: First iterations have biased estimates

❌ **Not scaling after dropout**: Must divide by (1 - p) during training

❌ **Wrong derivative**: sigmoid'(x) = σ(x)(1-σ(x)), not just σ'(x)

❌ **Batch norm in wrong mode**: Use training=False during inference

---

**Next**: [CNN Architectures →](07-CNN-ARCHITECTURES.md)


