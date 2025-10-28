# Interview Questions

## Table of Contents
1. [LLM & Transformers](#llm--transformers)
2. [RAG & Retrieval](#rag--retrieval)
3. [Deep Learning](#deep-learning)
4. [Core ML](#core-ml)
5. [System Design](#system-design)
6. [Behavioral Questions](#behavioral-questions)

---

## LLM & Transformers

**Q1: Explain self-attention from scratch.**

**Answer**:
1. Transform input into Query, Key, Value matrices
2. Compute attention scores: `Q @ K.T / sqrt(d_k)`
3. Apply softmax to get attention weights
4. Weighted sum: `weights @ V`
5. Allows each token to attend to all others

**Q2: What's the difference between GPT and BERT?**

**Answer**:

| Aspect | GPT | BERT |
|--------|-----|------|
| **Architecture** | Decoder-only | Encoder-only |
| **Training** | Causal LM (left-to-right) | Masked LM (bidirectional) |
| **Use Case** | Generation | Understanding |
| **Example** | Text completion, chatbots | Classification, NER |

**Q3: How does LoRA work? Why effective?**

**Answer**:
- **Method**: Freeze pretrained weights, add low-rank matrices $\Delta W = BA$
- **Params**: $d \times k$ → $r(d+k)$ where $r << \min(d,k)$
- **Savings**: For $d=k=4096, r=8$: 99.8% reduction
- **Why**: Most of fine-tuning can be done in low-rank space

**Q4: Explain transformer attention formula.**

**Answer**: `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V`

- `QK^T`: Measures similarity between queries and keys
- `/ sqrt(d_k)`: Scales to prevent tiny gradients
- `softmax`: Converts to probabilities
- `@ V`: Weighted sum of values

---

## RAG & Retrieval

**Q1: Design a RAG system for technical documentation.**

**Answer**:
1. **Indexing**:
   - Chunk by sections/headers
   - Embed with sentence-transformer
   - Store in ChromaDB/Pinecone
2. **Retrieval**:
   - Hybrid search (dense + BM25)
   - Retrieve top 100, rerank to 10
3. **Generation**:
   - Prompt with context + query
   - Use GPT-4 for answer
4. **Monitoring**:
   - Retrieval relevance
   - Answer quality (LLM-as-judge)
   - Latency tracking

**Q2: How choose chunk size?**

**Answer**:
- **Trade-off**: Large chunks → more context, less precise
- **Recommendation**:
  - Short docs: 200-400 tokens
  - Long docs: 800-1200 tokens
  - Overlap: 50-100 tokens
- **Test**: Try different sizes, measure retrieval accuracy

**Q3: Handle conflicting information across chunks?**

**Answer**:
1. **Rerank**: Use cross-encoder for relevance
2. **Majority vote**: If multiple chunks agree
3. **Confidence scores**: Weight by similarity
4. **Human review**: Flag low-confidence cases

---

## Deep Learning

**Q1: Why vanishing gradient problem in RNNs?**

**Answer**:
- Gradients computed via chain rule: $\frac{\partial L}{\partial W} = \prod \frac{\partial h_t}{\partial h_{t-1}}$
- Each step multiplies by `tanh'(z) ≤ 1`
- After 50 steps: $0.25^{50} ≈ 10^{-15}$ (essentially zero)
- **Solution**: LSTM gates, ResNet skip connections

**Q2: Explain backpropagation for 2-layer network.**

**Answer**:
```python
# Forward
z1 = X @ W1 + b1
a1 = ReLU(z1)
z2 = a1 @ W2 + b2
loss = MSE(y, z2)

# Backward
dz2 = y - z2  # Output gradient
dW2 = a1.T @ dz2
da1 = dz2 @ W2.T
dz1 = da1 * (z1 > 0)  # ReLU derivative
dW1 = X.T @ dz1
```

**Q3: Why batch normalization helps?**

**Answer**:
1. **Covariate shift**: Activations shift during training
2. **Normalization**: Keeps activations centered (mean=0, var=1)
3. **Benefits**:
   - Stable gradients
   - Higher learning rates
   - Less sensitive to initialization
   - Acts as regularizer

---

## Core ML

**Q1: When use decision tree vs SVM vs neural network?**

**Answer**:

| Data | Tree | SVM | NN |
|------|------|-----|----|
| **Small, interpretable** | ✅ | ✅ | ❌ |
| **Large, complex** | ⚠️ | ❌ | ✅ |
| **Non-linear** | ✅ | ✅ (with kernel) | ✅ |
| **Feature interactions** | ✅ | ⚠️ | ✅ |

**Q2: Explain bias-variance tradeoff.**

**Answer**:
- **Bias**: How far off average prediction
- **Variance**: How much predictions vary
- **Tradeoff**: Lower bias → Higher variance (overfitting)
- **Solution**: Regularization (L1/L2), ensemble methods

**Q3: Why ensemble methods work?**

**Answer**:
- **Diversity**: Different models make different errors
- **Averaging**: Reduces variance
- **Types**:
  - Bagging (Random Forest): Parallel, reduces variance
  - Boosting: Sequential, reduces bias
  - Stacking: Learn how to combine

---

## System Design

**Q1: Design real-time recommendation system.**

**Answer**:
```
1. Data: User behavior stream (Kafka)
2. Features: Real-time aggregation (Flink)
3. Training: Online learning + batch updates
4. Serving: Low-latency API (<50ms)
5. A/B Testing: 10% traffic to new model
6. Monitoring: Click-through rate, latency p99
```

**Q2: Scale LLM serving to 10K QPS.**

**Answer**:
```
1. Architecture:
   - Load balancer (nginx)
   - 50+ workers (GPUs)
   - Redis cache (common queries)
   - Message queue (request buffering)

2. Optimizations:
   - KV-cache (avoid recompute)
   - Batching (amortize overhead)
   - Quantization (INT8)
   - Model parallelism

3. Cost: ~$50K/month (estimate)
```

**Q3: Handle model versioning and rollback.**

**Answer**:
1. **Registry**: MLflow tracks versions
2. **Canary**: Deploy 5% traffic → monitor
3. **Rollback**: Auto-revert if metrics degrade
4. **Blue-Green**: Run two versions, switch traffic
5. **Monitoring**: p95 latency, error rate

---

## Behavioral Questions

**Q1: Tell me about a challenging ML project.**

**Structure**:
1. **Situation**: Context
2. **Task**: Goal
3. **Action**: What you did
4. **Result**: Outcome + metrics

**Example**:
"I built a fraud detection system. Challenge: Class imbalance (1:1000). I used SMOTE for oversampling, adjusted class weights, and ensemble methods. Result: 95% precision, 80% recall."

**Q2: How do you stay current with AI/ML?**

**Answer**:
- Papers: arXiv, Twitter (AI researchers)
- Conferences: NeurIPS, ICML
- Hands-on: Personal projects, competitions
- Communities: Reddit r/MachineLearning

**Q3: Describe a time you had to explain ML to non-technical audience.**

**Answer**:
"I presented our churn prediction model to stakeholders. I used simple analogies (like doctor making diagnosis based on symptoms), visualizations, and focused on business impact rather than technical details. Got buy-in to deploy."

---

## Quick Reference

### Must-Know Formulas

1. **Self-Attention**: `softmax(QK^T / sqrt(d_k))V`
2. **Cross-Entropy**: `-Σ y log(ŷ)`
3. **Softmax**: `exp(x_i) / Σ exp(x_j)`
4. **Sigmoid**: `1 / (1 + e^-x)`
5. **MSE**: `mean((y - ŷ)²)`
6. **Adam**: `θ ← θ - η * m̂ / (√v̂ + ε)`

### Key Concepts

1. **Overfitting**: Model memorizes training data
2. **Underfitting**: Model too simple
3. **Regularization**: Penalize complexity
4. **Hyperparameter**: Config, not learned
5. **Hyperparameter**: Model config

---

## Interview Tips

1. **Clarify requirements**: Don't assume
2. **Think out loud**: Show reasoning
3. **Trade-offs**: Acknowledge compromises
4. **Ask questions**: Show interest
5. **Stay calm**: It's okay to be unsure

---

**Good luck with your interview! 🎉**


