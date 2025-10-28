# RNN and Sequence Processing

## Table of Contents
1. [RNN Fundamentals](#rnn-fundamentals)
2. [LSTM Networks](#lstm-networks)
3. [GRU Networks](#gru-networks)
4. [Sequence-to-Sequence](#sequence-to-sequence)
5. [Attention Mechanisms](#attention-mechanisms)
6. [Modern Alternatives](#modern-alternatives)
7. [Interview Insights](#interview-insights)

---

## RNN Fundamentals

### What are RNNs?

**Definition**: Neural networks designed to process sequential data by maintaining **hidden state** across time steps.

**Key Idea**: Share parameters across time steps (unlike feedforward).

### Basic RNN Architecture

```
Input Sequence: x₁ → x₂ → x₃ → ... → xₜ
                   ↓     ↓     ↓
Hidden States:   h₁ → h₂ → h₃ → ... → hₜ
                   ↓     ↓     ↓
Outputs:         y₁    y₂    y₃         yₜ
```

### Mathematical Formulation

**At time step $t$**:

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b_h)$$

$$y_t = W_o h_t + b_o$$

Where:
- $x_t$ = input at time $t$
- $h_t$ = hidden state at time $t$
- $W_h, W_x, W_o$ = weight matrices
- $b_h, b_o$ = bias vectors

**Unrolled View**:

```
    x₁        x₂        x₃
     ↓         ↓         ↓
  ┌─────┐   ┌─────┐   ┌─────┐
  │  Wₓ │   │  Wₓ │   │  Wₓ │
  └──┬──┘   └──┬──┘   └──┬──┘
     │         │         │
  h₀ ┼→ h₁ ────┼→ h₂ ────┼→ h₃
     │         │         │
  ┌──┴──┐   ┌──┴──┐   ┌──┴──┐
  │  Wₕ │   │  Wₕ │   │  Wₕ │
  └─────┘   └─────┘   └─────┘
```

### Types of RNNs

#### 1. Many-to-One

**Input**: Sequence  
**Output**: Single prediction

```
Example: Sentiment classification
x₁ → x₂ → x₃ → y (Positive/Negative)
```

#### 2. One-to-Many

**Input**: Single value  
**Output**: Sequence

```
Example: Image captioning
x (image) → y₁ → y₂ → y₃
```

#### 3. Many-to-Many (Equal Length)

**Input**: Sequence  
**Output**: Sequence of same length

```
Example: Named Entity Recognition
x₁ → y₁, x₂ → y₂, x₃ → y₃
```

#### 4. Many-to-Many (Different Lengths)

**Input**: Sequence  
**Output**: Sequence of different length

```
Example: Machine translation
[He, loves, cats] → [Il, aime, les, chats]
```

### Implementation

```python
class SimpleRNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W_xh = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W_hy = np.random.randn(hidden_dim, output_dim) * 0.01
        
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)
    
    def forward(self, X):
        """
        X: [seq_len, input_dim]
        """
        seq_len = X.shape[0]
        h = np.zeros(self.W_hh.shape[0])  # Initial hidden state
        outputs = []
        
        for t in range(seq_len):
            # Update hidden state
            h = np.tanh(
                X[t] @ self.W_xh + h @ self.W_hh + self.b_h
            )
            
            # Compute output
            y_t = h @ self.W_hy + self.b_y
            outputs.append(y_t)
        
        return np.array(outputs)  # [seq_len, output_dim]
    
    def backward(self, X, targets, learning_rate=0.01):
        """
        Backpropagation Through Time (BPTT)
        """
        seq_len = X.shape[0]
        
        # Forward pass (store all activations)
        h_prev = np.zeros(self.W_hh.shape[0])
        hs = []  # Hidden states at each time
        
        for t in range(seq_len):
            h = np.tanh(X[t] @ self.W_xh + h_prev @ self.W_hh + self.b_h)
            hs.append(h)
            h_prev = h
        
        # Backward pass
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        dh_next = np.zeros(self.W_hh.shape[0])
        
        for t in reversed(range(seq_len)):
            # Gradient from output
            dy = outputs[t] - targets[t]
            dW_hy += hs[t].reshape(-1, 1) @ dy.reshape(1, -1)
            db_y += dy
            
            # Gradient to hidden
            dh = dy @ self.W_hy.T + dh_next
            
            # Gradient through tanh
            dh_raw = dh * (1 - hs[t]**2)
            
            # Gradients
            dW_xh += X[t].reshape(-1, 1) @ dh_raw.reshape(1, -1)
            dW_hh += hs[t-1].reshape(-1, 1) @ dh_raw.reshape(1, -1) if t > 0 else 0
            db_h += dh_raw
            
            dh_next = dh_raw @ self.W_hh.T
        
        # Update weights
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y
```

### Vanishing Gradient Problem

**Problem**: Gradients shrink exponentially as they propagate back through time.

**Why?**: Repeated multiplication of derivatives < 1 (e.g., tanh'(z) ≤ 1).

```
Gradient at t=50:
∂L/∂W ∝ (tanh')⁵⁰ ≈ 0.25⁵⁰ ≈ 10⁻¹⁵
```

**Impact**:
- Long-range dependencies lost
- Early time steps don't learn

**Solution**: Use gated architectures (LSTM, GRU).

---

## LSTM Networks

### Core Idea

**Long Short-Term Memory**: Use **gates** to control information flow.

### LSTM Cell

**Key Components**:

1. **Forget Gate**: What to discard
2. **Input Gate**: What new info to store
3. **Cell State**: Long-term memory
4. **Output Gate**: What to output

### Architecture (Single Step)

```
Input: xₜ, hₜ₋₁, Cₜ₋₁

           ┌─────────────┐
           │  Forget     │
           │     fₜ      │ ← Controls what to forget
           └─────────────┘
                 ↓
           ┌─────────────┐
           │  Input Gate  │
           │      iₜ      │ ← Controls new info
           └─────────────┘
                 ↓
           ┌─────────────┐
           │ Candidate  │
           │ Value C̃ₜ     │ ← New candidate values
           └─────────────┘
                 ↓
       ┌─────────────────────┐
       │   Cell State Cₜ    │
       │    (memory)         │
       └─────────────────────┘
                 ↓
           ┌─────────────┐
           │ Output Gate │
           │      oₜ     │ ← Controls output
           └─────────────┘
                 ↓
Output: hₜ, Cₜ
```

### Mathematical Formulation

**Step 1: Forget Gate**

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

Decides what information to discard from cell state.

**Step 2: Input Gate**

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

Decides what new information to store.

**Step 3: Update Cell State**

$$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$

Combines old state (filtered by forget) + new info (filtered by input).

**Step 4: Output Gate**

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t * \tanh(C_t)$$

Decides what parts of cell state to output.

Where:
- $\sigma$ = sigmoid (gates output values in [0,1])
- $*$ = element-wise multiplication
- $\cdot$ = matrix multiplication

### Implementation

```python
class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        # All gates share input and hidden state
        # Gate parameters: W_f, W_i, W_C, W_o
        # Each: [input_dim + hidden_dim, hidden_dim]
        
        dim = input_dim + hidden_dim
        self.W_f = np.random.randn(dim, hidden_dim) * 0.01
        self.W_i = np.random.randn(dim, hidden_dim) * 0.01
        self.W_C = np.random.randn(dim, hidden_dim) * 0.01
        self.W_o = np.random.randn(dim, hidden_dim) * 0.01
        
        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_C = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)
    
    def forward(self, x, h_prev, C_prev):
        """
        x: [batch, input_dim]
        h_prev: [batch, hidden_dim]
        C_prev: [batch, hidden_dim]
        """
        # Concatenate
        concat = np.concatenate([h_prev, x], axis=1)  # [batch, input_dim + hidden_dim]
        
        # Forget gate
        f_t = sigmoid(concat @ self.W_f + self.b_f)
        
        # Input gate
        i_t = sigmoid(concat @ self.W_i + self.b_i)
        
        # Candidate values
        C_tilde = np.tanh(concat @ self.W_C + self.b_C)
        
        # Update cell state
        C_t = f_t * C_prev + i_t * C_tilde
        
        # Output gate
        o_t = sigmoid(concat @ self.W_o + self.b_o)
        
        # Hidden state
        h_t = o_t * np.tanh(C_t)
        
        return h_t, C_t
```

### Why LSTM Works

**Gradient Flow**:

$$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$

**Derivative**:

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

Since $f_t \in (0, 1)$, gradients controlled by forget gate. Can learn $f_t = 1$ to preserve gradient for many steps.

### Bidirectional LSTM

**Process sequence in both directions**:

```
Forward:  x₁ → x₂ → x₃ → h₃
Backward: x₁ ← x₂ ← x₃ ← h₃

Output: h₃ = concat[h₃_forward, h₃_backward]
```

```python
class BidirectionalLSTM:
    def __init__(self, input_dim, hidden_dim):
        self.lstm_forward = LSTM(input_dim, hidden_dim)
        self.lstm_backward = LSTM(input_dim, hidden_dim)
    
    def forward(self, X):
        """
        X: [seq_len, batch, input_dim]
        """
        # Forward pass
        h_forward = self.lstm_forward(X)
        
        # Backward pass
        h_backward = self.lstm_backward(reverse(X))
        
        # Concatenate
        h = concatenate([h_forward, h_backward], dim=-1)
        
        return h  # [seq_len, batch, 2*hidden_dim]
```

---

## GRU Networks

### Core Idea

**Gated Recurrent Unit**: **Simplified LSTM** (same concept, fewer gates).

**Key Difference**: Combines forget and input gates into **update gate**.

### Architecture

```
Input: xₜ, hₜ₋₁

      ┌─────────┐
      │ Update  │ ← Controls what to remember/forget
      │  Gate zₜ│
      └─────────┘
            ↓
      ┌─────────┐
      │Reset    │ ← Controls relevance of previous state
      │Gate rₜ  │
      └─────────┘
            ↓
      ┌─────────┐
      │Candidate│
      │  h̃ₜ    │
      └─────────┘
            ↓
       hₜ (output)
```

### Mathematical Formulation

**Step 1: Update Gate**

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$

**Step 2: Reset Gate**

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$

**Step 3: Candidate Activation**

$$\tilde{h}_t = \tanh(W \cdot [r_t * h_{t-1}, x_t])$$

**Step 4: Update Hidden State**

$$h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t$$

Where:
- $z_t$ = update gate (how much new info vs old)
- $r_t$ = reset gate (how relevant previous state)
- $(1 - z_t)$ = how much to keep from previous
- $z_t$ = how much new info to incorporate

### Comparison: LSTM vs GRU

| Aspect | LSTM | GRU |
|--------|------|-----|
| **Gates** | 3 (forget, input, output) | 2 (update, reset) |
| **Cell State** | Yes (separate) | No (combined with hidden) |
| **Parameters** | More | Fewer (~33% less) |
| **Speed** | Slower | Faster |
| **Memory** | Better long-term | Good short-medium |
| **Use Case** | Very long sequences | Most tasks |

### Implementation

```python
class GRUCell:
    def __init__(self, input_dim, hidden_dim):
        dim = input_dim + hidden_dim
        
        self.W_z = np.random.randn(dim, hidden_dim) * 0.01  # Update gate
        self.W_r = np.random.randn(dim, hidden_dim) * 0.01  # Reset gate
        self.W_h = np.random.randn(dim, hidden_dim) * 0.01  # Candidate
        
        self.b_z = np.zeros(hidden_dim)
        self.b_r = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)
    
    def forward(self, x, h_prev):
        concat = np.concatenate([h_prev, x], axis=1)
        
        # Update gate
        z_t = sigmoid(concat @ self.W_z + self.b_z)
        
        # Reset gate
        r_t = sigmoid(concat @ self.W_r + self.b_r)
        
        # Candidate
        h_tilde = np.tanh(
            concat @ self.W_h + self.b_h,
            # But W_h needs modified input
            # Typically: W_h @ [r_t * h_prev, x]
        )
        
        # This is simplified - need to properly handle reset gate
        # Full implementation would separate W_h into two matrices
        
        # Update
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t
```

---

## Sequence-to-Sequence

### Encoder-Decoder Architecture

**Purpose**: Map variable-length input → variable-length output.

**Applications**: Machine translation, summarization.

### Architecture

```
Encoder (RNN/LSTM)
═══════════════════
Input: "How are you?"
    ↓
Embedding
    ↓
LSTM → LSTM → LSTM → LSTM
    ↓
Hidden State (Context Vector)

Decoder (RNN/LSTM)
═══════════════════
Context Vector
    ↓
Hidden State
    ↓
LSTM → LSTM → LSTM → LSTM
    ↓
"Comment allez-vous?"
```

### Implementation

```python
class EncoderDecoder:
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        self.encoder = LSTM(vocab_size, embed_dim, hidden_dim)
        self.decoder = LSTM(vocab_size, embed_dim, hidden_dim)
    
    def encode(self, X):
        """
        X: [seq_len, batch, vocab_size]
        """
        # Forward through encoder
        _, h_last = self.encoder(X)
        
        # Context vector
        context = h_last
        return context
    
    def decode(self, context, max_len=50):
        """
        Generate sequence from context
        """
        # Start with [BOS] token
        output = [bos_token]
        h = context
        
        for t in range(max_len):
            # Embed current output
            x_t = embed(output[-1])
            
            # Decoder step
            y_t, h = self.decoder.step(x_t, h)
            
            # Get token
            token = argmax(y_t)
            output.append(token)
            
            # Stop at [EOS]
            if token == eos_token:
                break
        
        return output
    
    def forward(self, X_source, Y_target):
        """
        Training: Teacher forcing
        """
        # Encode source
        context = self.encode(X_source)
        
        # Decode with ground truth
        # (Use Y_target as inputs to decoder, not own predictions)
        outputs = self.decoder(Y_target, context)
        
        return outputs
```

### Attention Mechanism

**Problem**: Bottleneck at context vector (tries to compress entire source).

**Solution**: **Attention** - dynamically focus on different parts of source.

**Attentive Encoder-Decoder**:

```python
class AttentiveDecoder:
    def __init__(self, encoder_dim, decoder_dim, hidden_dim):
        self.W_a = Linear(encoder_dim + decoder_dim, hidden_dim)
        self.v = Linear(hidden_dim, 1)
    
    def forward(self, encoder_outputs, decoder_hidden):
        """
        encoder_outputs: [seq_len, batch, encoder_dim]
        decoder_hidden: [batch, decoder_dim]
        """
        seq_len = encoder_outputs.shape[0]
        
        # Compute attention scores
        scores = []
        for i in range(seq_len):
            # Combine
            combined = concatenate([encoder_outputs[i], decoder_hidden], dim=-1)
            
            # Score
            score = self.v(tanh(self.W_a(combined)))
            scores.append(score)
        
        scores = stack(scores)  # [seq_len, batch, 1]
        
        # Softmax
        attention_weights = softmax(scores, dim=0)
        
        # Weighted sum
        context = sum(
            attention_weights[i] * encoder_outputs[i]
            for i in range(seq_len)
        )  # [batch, encoder_dim]
        
        return context, attention_weights
```

---

## Modern Alternatives

### Why RNNs are Less Common

**Limitations**:
- Sequential processing (can't parallelize)
- Still have vanishing gradient issues (even with gates)
- Difficulty capturing long-range dependencies

**Modern Replacements**:
- **Transformers**: Attention-based (parallel, long-range)
- **WaveNet**: Convolutional (dilated convolutions)
- **Temporal Convolutional Networks**: CNNs for sequences

### When to Use RNNs/LSTMs

✅ **Still Good For**:
- Real-time streaming data
- Online learning (process as data arrives)
- When order matters and length varies
- Simpler architectures

❌ **Not Ideal For**:
- Very long sequences
- Parallel processing requirements
- SOTA benchmarks (use Transformers)

---

## Interview Insights

### Common Questions

**Q1: Explain LSTM vs RNN.**

**Answer**: RNN has hidden state passed through time, but suffers vanishing gradient. LSTM adds cell state (long-term memory) + gates (forget, input, output) to control information flow. Gradient flows through cell state can persist for many steps, enabling long-range dependencies.

**Q2: What causes vanishing gradients in RNNs?**

**Answer**: Repeated multiplication of derivatives. Each step multiplies by tanh'(z) ≤ 1. After many steps, gradient becomes exponentially small. Early time steps barely update.

**Q3: How does attention fix sequence-to-sequence?**

**Answer**: Instead of single context vector, attention allows decoder to focus on different encoder states at each step. Creates dynamic context, no information bottleneck.

**Q4: GRU vs LSTM trade-offs?**

**Answer**: GRU simpler (2 gates vs 3), faster, fewer parameters. LSTM has separate cell state for longer memory. GRU often similar performance with less compute.

### Common Pitfalls

❌ **Saying GRU has no forget mechanism**: Update gate handles this

❌ **Ignoring memory complexity**: LSTMs expensive for long sequences

❌ **Not mentioning parallelization issue**: RNNs sequential by design

---

**Next**: [Core ML Algorithms →](09-CORE-ML-ALGORITHMS.md)

