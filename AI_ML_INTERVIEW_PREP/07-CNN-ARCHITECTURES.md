# CNN Architectures

## Table of Contents
1. [Convolution Fundamentals](#convolution-fundamentals)
2. [Pooling Operations](#pooling-operations)
3. [Modern Architectures](#modern-architectures)
4. [Vision Transformers](#vision-transformers)
5. [Transfer Learning](#transfer-learning)
6. [Interview Insights](#interview-insights)

---

## Convolution Fundamentals

### What is Convolution?

**Definition**: Element-wise multiplication and summation between a kernel (filter) and input.

### 2D Convolution

```
Input Image (6×6)          Kernel (3×3)
┌─┬─┬─┬─┬─┬─┐           ┌─┬─┬─┐
│4│2│1│3│5│7│           │1│0│-1│
├─┼─┼─┼─┼─┼─┤   *       │1│0│-1│
│3│6│2│4│1│8│           │1│0│-1│
├─┼─┼─┼─┼─┼─┤           └─┴─┴─┘
│7│1│4│2│6│3│
├─┼─┼─┼─┼─┼─┤
│5│3│7│1│2│4│
├─┼─┼─┼─┼─┼─┤
│2│4│6│8│3│5│
└─┴─┴─┴─┴─┴─┘

Step 1: Place kernel on top-left
         ┌─┬─┬─┐
         │4│2│1│
         │3│6│2│
         │7│1│4│
         └─┴─┴─┘

Result: 4×1 + 2×0 + 1×(-1) + 3×1 + 6×0 + 2×(-1) + 7×1 + 1×0 + 4×(-1)
      = 4 - 1 + 3 - 2 + 7 - 4 = 7
```

### Mathematical Definition

For input $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$ and kernel $\mathbf{K} \in \mathbb{R}^{k \times k}$:

$$(\mathbf{X} * \mathbf{K})_{i,j} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \mathbf{X}_{i+m, j+n} \cdot \mathbf{K}_{m,n}$$

### Key Concepts

#### 1. Stride

**Distance** kernel moves each step.

```python
def conv2d_with_stride(input, kernel, stride=1, padding=0):
    """
    input: [batch, C, H, W]
    kernel: [C_out, C_in, k, k]
    """
    batch, C_in, H, W = input.shape
    C_out, _, k, _ = kernel.shape
    
    # Calculate output dimensions
    H_out = (H + 2*padding - k) // stride + 1
    W_out = (W + 2*padding - k) // stride + 1
    
    output = zeros([batch, C_out, H_out, W_out])
    
    for i in range(H_out):
        for j in range(W_out):
            # Extract patch
            start_i = i * stride
            start_j = j * stride
            end_i = start_i + k
            end_j = start_j + k
            
            patch = input[:, :, start_i:end_i, start_j:end_j]
            
            # Convolve
            output[:, :, i, j] = sum(patch * kernel, axis=(2, 3))
    
    return output
```

**Example**:
```
Input: 28×28, Kernel: 3×3
Stride 1: Output = 26×26  (28-3+1)
Stride 2: Output = 13×13  ((28-3)/2+1)
```

#### 2. Padding

**Add zeros** around input to control output size.

```
Input (5×5)     With Padding 1
┌─┬─┬─┬─┬─┐     0 0 0 0 0 0 0
│1│2│3│4│5│     0 1 2 3 4 5 0
│2│3│4│5│6│     0 2 3 4 5 6 0
│3│4│5│6│7│     0 3 4 5 6 7 0
│4│5│6│7│8│     0 4 5 6 7 8 0
│5│6│7│8│9│     0 5 6 7 8 9 0
└─┴─┴─┴─┴─┘     0 0 0 0 0 0 0
```

**Common Padding Types**:

- **Valid**: No padding, output smaller
- **Same**: Padding to keep same size

$$\text{Output Size} = \frac{\text{Input Size} + 2p - k}{\text{stride}} + 1$$

For **same padding**: $p = \frac{k-1}{2}$

#### 3. Channels (Depth)

**Multiple kernels** learn different features.

```
Input: [H, W, C_in]
Kernel: [C_out, C_in, k, k]
Output: [H', W', C_out]
```

**Example**: Edge detection vs texture detection

```python
def conv2d_multi_channel(input, kernel, stride=1, padding=0):
    """
    input: [batch, C_in, H, W]
    kernel: [C_out, C_in, k, k]
    """
    batch, C_in, H, W = input.shape
    C_out, _, k, k = kernel.shape
    
    # Add padding
    if padding > 0:
        input = pad(input, padding)
        H += 2*padding
        W += 2*padding
    
    H_out = (H - k) // stride + 1
    W_out = (W - k) // stride + 1
    
    output = zeros([batch, C_out, H_out, W_out])
    
    for b in range(batch):
        for c_out in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    # Extract patch
                    h_start = h * stride
                    w_start = w * stride
                    patch = input[b, :, h_start:h_start+k, w_start:w_start+k]
                    
                    # Convolve (sum over all input channels)
                    output[b, c_out, h, w] = sum(
                        patch * kernel[c_out, :, :, :]
                    )
    
    return output
```

### Convolution vs Fully Connected

| Aspect | FC Layer | Convolution |
|--------|----------|------------|
| **Params** | H×W×C | k×k×C (fixed) |
| **Translation** | Not invariant | Invariant |
| **Spatial info** | Lost | Preserved |

**Example** (224×224×3 image):
- **FC**: 224×224×3 × 1000 = 150M params
- **Conv**: 3×3×3 = 27 params per output

---

## Pooling Operations

### Purpose

**Reduce spatial dimensions**, maintain depth.

### Max Pooling

**Takes maximum** in each window.

```
Input (4×4)        Output (2×2)
┌─┬─┬─┬─┐          ┌─┬─┐
│1│2│3│4│          │4│8│    max(1,2,5,6)=4
├─┼─┼─┼─┤  →       ├─┼─┤    max(3,4,7,8)=8
│5│6│7│8│          │6│9│    max(9,10,13,14)=6
├─┼─┼─┼─┤          └─┴─┘    max(11,12,15,16)=9
│9│10│11│12│
├─┼─┼─┼─┤
│13│14│15│16│
└─┴─┴─┴─┘
```

**Formula**:

$$\text{MaxPool}(\mathbf{X})_{i,j} = \max_{m=0}^{p-1} \max_{n=0}^{p-1} \mathbf{X}_{i·s+m, j·s+n}$$

Where $p$ = pool size, $s$ = stride.

### Average Pooling

**Takes average** in each window.

```python
def max_pool2d(input, pool_size=2, stride=2):
    """
    input: [batch, C, H, W]
    """
    batch, C, H, W = input.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    output = zeros([batch, C, H_out, W_out])
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            h_end = h_start + pool_size
            w_end = w_start + pool_size
            
            patch = input[:, :, h_start:h_end, w_start:w_end]
            output[:, :, i, j] = patch.max(axis=(2, 3))  # Max across spatial dims
    
    return output

def avg_pool2d(input, pool_size=2, stride=2):
    """Average pooling"""
    # Same as max_pool, but use .mean() instead of .max()
    ...
```

### Why Pooling?

**Benefits**:
1. **Invariance**: Small translations don't affect output
2. **Reduce params**: Smaller feature maps
3. **Increase receptive field**: See larger context
4. **Prevent overfitting**: Downsampling

**Drawbacks**:
- Loses spatial information
- Small objects might disappear

---

## Modern Architectures

### LeNet-5 (1998)

**Classic** handwritten digit recognition.

```
Input (32×32) → Conv (6@28×28) → Pool → Conv (16@10×10) → Pool → FC → Output
```

### AlexNet (2012)

**Innovations**: ReLU, Dropout, Data augmentation.

```
Input (224×224×3)
    ↓
Conv 1: 96@55×55 (stride 4)
    ↓
MaxPool 3×3
    ↓
Conv 2: 256@27×27
    ↓
MaxPool 3×3
    ↓
Conv 3: 384@13×13
    ↓
Conv 4: 384@13×13
    ↓
Conv 5: 256@13×13
    ↓
MaxPool 3×3
    ↓
FC 4096 → FC 4096 → FC 1000
    ↓
Output (1000 classes)
```

### VGG (2014)

**Key**: Small filters, deeper network.

**Principle**: Replace large filters with stack of small filters.

```
3×3 conv vs 5×5 conv
Params: 3×3×2 = 18 vs 25 (fewer!)
Non-linearity: 2 vs 1 (more expressive)
```

**VGG-16 Architecture**:
```
Input (224×224×3)
2× Conv(64) → Pool
2× Conv(128) → Pool
3× Conv(256) → Pool
3× Conv(512) → Pool
3× Conv(512) → Pool
FC 4096 → FC 4096 → FC 1000
```

### ResNet (2015) - Residual Networks

**Problem**: Deeper networks harder to train (vanishing gradients).

**Solution**: **Skip connections**.

```
    x
    │
    ├─────────┐
    ↓         │
  Conv 3×3    │
    ↓         │
  BatchNorm   │
    ↓         │
  ReLU        │
    ↓         │
  Conv 3×3    │
    ↓         │
  BatchNorm   │
    ↓         │
    + ←───────┘
    ↓
  ReLU
    ↓
   out
```

**Formula**:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

**Benefits**:
- Enables 100+ layer networks
- Identity mapping allows gradient flow
- Easier optimization

```python
class ResidualBlock:
    def __init__(self, C_in, C_out, stride=1):
        self.conv1 = Conv2d(C_in, C_out, 3, stride, padding=1)
        self.bn1 = BatchNorm2d(C_out)
        self.conv2 = Conv2d(C_out, C_out, 3, padding=1)
        self.bn2 = BatchNorm2d(C_out)
        
        # If dimensions change, need projection
        if C_in != C_out or stride != 1:
            self.shortcut = Conv2d(C_in, C_out, 1, stride)
        else:
            self.shortcut = identity
    
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection
        if self.shortcut is not identity:
            identity = self.shortcut(x)
        
        out = out + identity
        out = relu(out)
        
        return out
```

**ResNet-50, ResNet-101, ResNet-152**

Bottleneck architecture:

```
Conv 1×1 (downsample) → Conv 3×3 → Conv 1×1 (upsample)
```

### Inception (2014)

**Idea**: Let network learn optimal kernel sizes.

**Inception Module**:
```
Input
    ├─ Conv 1×1 ──→ Concatenate
    ├─ Conv 1×1 → Conv 3×3 ──┤
    ├─ Conv 1×1 → Conv 5×5 ──┤
    └─ MaxPool → Conv 1×1 ────┘
```

**Benefits**:
- Multi-scale features
- Sparse connections (efficient)
- GoogLeNet won ILSVRC 2014

### DenseNet (2017)

**Idea**: Each layer connected to every previous layer.

```
Input → L1 → x₁ ─┐
  └───────────────┼─→ L2 → x₂ ─┐
                  └───────────┼─→ L3 → x₃
                             └───────────...
```

**Formula**:

$$\mathbf{x}_l = \mathcal{H}_l([\mathbf{x}_0, \mathbf{x}_1, ..., \mathbf{x}_{l-1}])$$

**Benefits**:
- Feature reuse
- Fewer parameters
- Stronger gradient flow

```python
class DenseBlock:
    def __init__(self, num_layers, growth_rate):
        self.layers = []
        for i in range(num_layers):
            self.layers.append(DenseLayer(growth_rate))
    
    def forward(self, x):
        features = [x]
        
        for layer in self.layers:
            # Input is concatenation of all previous features
            new_features = layer(concatenate(features, dim=1))
            features.append(new_features)
        
        return concatenate(features, dim=1)
```

---

## Vision Transformers

### From CNN to ViT

**Traditional**: CNN backbone

**ViT**: Treat image as sequence of patches

### Patch Embedding

```
Image (224×224×3)
    ↓
Split into patches (16×16 each)
    ↓
14×14 = 196 patches
    ↓
Flatten each: 16×16×3 = 768
    ↓
[CLS] token + Patch embeddings
    ↓
Position embeddings
    ↓
Transformer Encoder
    ↓
Classification head
```

```python
class PatchEmbedding:
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Learnable projection
        self.proj = Linear(patch_size**2 * 3, embed_dim)
        
        # Position embeddings
        self.pos_embed = learnable_embedding(self.num_patches + 1, embed_dim)
    
    def forward(self, x):
        # x: [batch, 3, 224, 224]
        batch = x.shape[0]
        
        # Reshape to patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # Now: [batch, 3, 14, 14, 16, 16]
        
        patches = patches.contiguous().view(batch, self.num_patches, -1)
        # Now: [batch, 196, 768]
        
        # Project
        patch_emb = self.proj(patches)
        
        # Add [CLS] token
        cls_token = self.cls_token.expand(batch, 1, -1)
        embeddings = concatenate([cls_token, patch_emb], dim=1)
        
        # Add position embeddings
        embeddings = embeddings + self.pos_embed
        
        return embeddings
```

### ViT Architecture

```python
class VisionTransformer:
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, 
                 num_layers=12, num_heads=12, mlp_dim=3072):
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim)
        
        # Transformer blocks
        self.blocks = []
        for i in range(num_layers):
            self.blocks.append(TransformerBlock(embed_dim, num_heads, mlp_dim))
        
        # Classification head
        self.head = Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [batch, 197, 768]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Extract [CLS] token
        cls = x[:, 0, :]  # [batch, 768]
        
        # Classify
        output = self.head(cls)
        
        return output
```

### ViT vs CNN

| Aspect | CNN | ViT |
|--------|-----|-----|
| **Inductive bias** | Translation equivariance | None |
| **Data needed** | Less | More |
| **Global attention** | Later layers | All layers |
| **Interpretability** | Filters | Attention maps |

---

## Transfer Learning

### Concept

**Use pretrained model** for new task.

### Process

1. **Pretraining**: Train on large dataset (ImageNet)
2. **Fine-tuning**: Adapt to specific task
3. **Feature extraction**: Use as frozen feature extractor

### Feature Extraction

```python
def feature_extraction(model, data_loader):
    """
    Use pretrained model as feature extractor
    """
    model.eval()
    features = []
    labels = []
    
    for x, y in data_loader:
        # Forward up to last layer
        with torch.no_grad():
            feat = model.features(x)
        
        features.append(feat)
        labels.append(y)
    
    # Train classifier on features
    features = concatenate(features)
    classifier = train_classifier(features, labels)
```

### Fine-Tuning Strategies

**1. Full fine-tuning**:
```python
for param in model.parameters():
    param.requires_grad = True
```

**2. Progressive unfreezing**:
```python
# Freeze all
for param in model.parameters():
    param.requires_grad = False

# Unfreeze top layers
for param in model.classifier.parameters():
    param.requires_grad = True

# After convergence, unfreeze more
for param in model.backbone.layer4.parameters():
    param.requires_grad = True
```

**3. Differential learning rates**:
```python
optimizer = Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-4},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
])
```

---

## Interview Insights

### Common Questions

**Q1: Explain convolution operation.**

**Answer**: Convolution slides a kernel/filter over input, computes element-wise multiplication and sums results. It's translation-invariant, parameter-efficient, and preserves spatial structure. Used to detect local patterns (edges, textures).

**Q2: Why ResNet successful?**

**Answer**: Skip connections (residual connections) solve vanishing gradient problem. Allows training very deep networks (100+ layers). Identity mapping ensures gradients flow through entire network.

**Q3: CNN vs Vision Transformer?**

**Answer**:

| Aspect | CNN | ViT |
|--------|-----|-----|
| **Inductive bias** | Translation equivariance, locality | Minimal |
| **Data efficiency** | Better with small datasets | Needs large datasets |
| **Global context** | Builds gradually | Immediate |
| **Interpretability** | Filter visualizations | Attention maps |
| **Training** | Faster | Slower (quadratic complexity) |
| **Best for** | Small datasets, most tasks | Large datasets, global features |

**Q4: How does pooling help?**

**Answer**:
1. **Dimensionality reduction**: Fewer parameters
2. **Translation invariance**: Small shifts don't matter
3. **Larger receptive field**: Each layer sees more context
4. **Overfitting reduction**: Less capacity

**Q5: Design a CNN for image classification.**

**Answer**:
```
Input (224×224×3)
    ↓
Conv(64, 7×7, stride=2) + BatchNorm + ReLU + Pool
    ↓
Residual block × 3 (64 filters)
    ↓
Residual block × 4 (128 filters)
    ↓
Residual block × 6 (256 filters)
    ↓
Residual block × 3 (512 filters)
    ↓
Global Average Pooling
    ↓
FC(1000 classes)
```

### Common Pitfalls

❌ **Wrong output size**: Forgetting stride/padding effects

❌ **Confusing convolution dimensions**: Shape is [C_out, C_in, k, k]

❌ **Not accounting for receptive field**: Each layer sees limited region

❌ **Pooling misuse**: Max pool good for sparse features, avg pool for dense

### Best Practices

1. **Start with pretrained models**: ResNet, EfficientNet
2. **Use batch normalization**: Stabilizes training
3. **Progressive resizing**: Train on smaller images first
4. **Data augmentation**: Rotation, crop, color jitter
5. **Learning rate schedule**: Warmup + decay

---

## Key Papers

1. **"ImageNet Classification with Deep Convolutional Neural Networks"** (Krizhevsky et al., 2012) - AlexNet
2. **"Very Deep Convolutional Networks"** (VGG, Simonyan & Zisserman, 2015)
3. **"Deep Residual Learning"** (He et al., 2016) - ResNet
4. **"Going Deeper"** (Szegedy et al., 2015) - Inception
5. **"An Image is Worth 16x16 Words"** (Dosovitskiy et al., 2020) - ViT

---

**Next**: [RNN and Sequences →](08-RNN-AND-SEQUENCES.md)


