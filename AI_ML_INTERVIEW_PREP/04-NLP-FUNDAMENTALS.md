# NLP Fundamentals

## Table of Contents
1. [Tokenization](#tokenization)
2. [Word Embeddings](#word-embeddings)
3. [Contextual Embeddings](#contextual-embeddings)
4. [Language Modeling](#language-modeling)
5. [Common NLP Tasks](#common-nlp-tasks)
6. [Interview Insights](#interview-insights)

---

## Tokenization

### What is Tokenization?

**Definition**: Breaking text into smaller units (tokens) for processing.

**Why Important**: Neural networks need fixed-size inputs, can't process raw text.

### Tokenization Levels

```
Text: "OpenAI's GPT-4 is amazing!"

Word-level:      ["OpenAI's", "GPT-4", "is", "amazing", "!"]
Subword-level:   ["Open", "AI", "'s", "G", "PT", "-", "4", "is", "amazing", "!"]
Character-level: ["O", "p", "e", "n", "A", "I", "'", "s", " ", ...]
```

| Level | Vocab Size | OOV Handling | Use Case |
|-------|------------|--------------|----------|
| **Word** | 50K-100K | Poor (unknown words) | Simple tasks |
| **Subword** | 30K-50K | Excellent | Modern LLMs |
| **Character** | 100-256 | Perfect | Language modeling |

---

## Tokenization Algorithms

### 1. Byte-Pair Encoding (BPE)

**Core Idea**: Iteratively merge most frequent character pairs.

**Algorithm**:

```python
def train_bpe(corpus, num_merges):
    """
    corpus: List of words with character splits
    num_merges: Number of merge operations
    """
    # Initialize: Split into characters
    vocab = set()
    word_freqs = count_words(corpus)
    
    # Initial vocab: all characters
    for word in word_freqs:
        for char in word:
            vocab.add(char)
    
    # Perform merges
    for i in range(num_merges):
        # Count all adjacent pairs
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for j in range(len(symbols) - 1):
                pair = (symbols[j], symbols[j+1])
                pair_freqs[pair] += freq
        
        # Find most frequent pair
        best_pair = max(pair_freqs, key=pair_freqs.get)
        
        # Merge the pair in all words
        word_freqs = merge_pair(word_freqs, best_pair)
        
        # Add merged token to vocab
        vocab.add(''.join(best_pair))
    
    return vocab

def merge_pair(word_freqs, pair):
    """Merge a pair in all words"""
    new_word_freqs = {}
    merged = ''.join(pair)
    
    for word, freq in word_freqs.items():
        new_word = word.replace(' '.join(pair), merged)
        new_word_freqs[new_word] = freq
    
    return new_word_freqs
```

**Example**:

```
Corpus: ["low", "lower", "lowest", "newer", "wider"]

Initial: l o w, l o w e r, l o w e s t, n e w e r, w i d e r

Step 1: Most frequent pair = (e, r) → merge to "er"
Result: l o w, l o w er, l o w e s t, n ew er, w i d er

Step 2: Most frequent pair = (l, o) → merge to "lo"
Result: lo w, lo w er, lo w e s t, n ew er, w i d er

Step 3: Most frequent pair = (lo, w) → merge to "low"
Result: low, low er, low e s t, n ew er, w i d er

Final vocab: {l, o, w, e, r, s, t, n, i, d, er, lo, low, ...}
```

**Encoding**:
```python
def bpe_encode(text, vocab, merges):
    """
    Encode text using learned BPE vocabulary
    """
    # Start with character-level
    tokens = list(text)
    
    # Apply merges in order
    for merge in merges:
        i = 0
        while i < len(tokens) - 1:
            if (tokens[i], tokens[i+1]) == merge:
                # Merge pair
                tokens[i:i+2] = [''.join(merge)]
            else:
                i += 1
    
    return tokens
```

**Used By**: GPT-2, GPT-3, RoBERTa

---

### 2. WordPiece

**Core Idea**: Similar to BPE, but chooses merges based on likelihood increase.

**Key Difference**: Selection criterion

**BPE**: Frequency-based
$$\text{score}(a, b) = \text{count}(ab)$$

**WordPiece**: Likelihood-based
$$\text{score}(a, b) = \frac{P(ab)}{P(a) \times P(b)}$$

**Algorithm**:

```python
def train_wordpiece(corpus, vocab_size):
    """
    corpus: List of words
    vocab_size: Target vocabulary size
    """
    # Initialize with characters
    vocab = set(char for word in corpus for char in word)
    
    while len(vocab) < vocab_size:
        # Compute likelihood scores for all pairs
        scores = {}
        for word in corpus:
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                if pair not in scores:
                    # Score = P(ab) / (P(a) * P(b))
                    scores[pair] = likelihood_score(pair, corpus)
        
        # Select best pair
        best_pair = max(scores, key=scores.get)
        
        # Add to vocabulary
        vocab.add(''.join(best_pair))
        
        # Update corpus representation
        corpus = merge_in_corpus(corpus, best_pair)
    
    return vocab

def likelihood_score(pair, corpus):
    """Compute P(ab) / (P(a) * P(b))"""
    a, b = pair
    ab = ''.join(pair)
    
    p_ab = count(ab, corpus) / total_tokens(corpus)
    p_a = count(a, corpus) / total_tokens(corpus)
    p_b = count(b, corpus) / total_tokens(corpus)
    
    return p_ab / (p_a * p_b)
```

**Special Tokens**:
- `##` prefix for subword continuations
- Example: "playing" → ["play", "##ing"]

**Used By**: BERT, DistilBERT

---

### 3. SentencePiece

**Core Idea**: Treat text as sequence of Unicode characters (language-agnostic).

**Key Features**:
- **No pre-tokenization**: Doesn't require word boundaries
- **Lossless**: Can perfectly reconstruct original text
- **Whitespace as token**: Treats spaces as `▁` character

**Algorithms**: Supports both BPE and Unigram

**Example**:
```
Text: "Hello world"

Tokenization: ["▁Hello", "▁world"]
              ↑ Space encoded as underscore

Detokenization: Remove ▁, get back "Hello world"
```

**Unigram Language Model**:

Instead of merging, starts with large vocab and prunes.

```python
def train_unigram(corpus, vocab_size):
    """
    Start with large vocabulary, iteratively remove tokens
    """
    # Initialize with all substrings
    vocab = initialize_large_vocab(corpus)
    
    while len(vocab) > vocab_size:
        # For each token, compute loss if removed
        losses = {}
        for token in vocab:
            vocab_minus_token = vocab - {token}
            losses[token] = compute_corpus_loss(corpus, vocab_minus_token)
        
        # Remove token with smallest loss increase
        token_to_remove = min(losses, key=losses.get)
        vocab.remove(token_to_remove)
    
    return vocab

def compute_corpus_loss(corpus, vocab):
    """Negative log-likelihood of corpus given vocab"""
    loss = 0
    for word in corpus:
        # Find best segmentation
        best_segmentation = viterbi(word, vocab)
        loss -= log_prob(best_segmentation)
    return loss
```

**Used By**: T5, ALBERT, XLNet, multilingual models

---

### Tokenization Best Practices

#### Handling Unknown Words

**Problem**: Word not in vocabulary.

**Solutions**:
1. **Subword tokenization**: Break into known pieces
2. **`<UNK>` token**: Replace with special token
3. **Character fallback**: Split to characters

#### Byte-Level BPE

**Enhancement**: Operate on bytes instead of characters.

**Benefits**:
- **Fixed base vocab**: 256 bytes (not thousands of chars)
- **Universal**: Works for any language/emoji
- **No `<UNK>`**: Every byte is in vocab

```python
# Convert text to bytes
text = "Hello 👋"
byte_seq = text.encode('utf-8')  # [72, 101, 108, 108, 111, 32, 240, 159, 145, 139]

# Apply BPE on byte sequences
tokens = bpe_encode(byte_seq, byte_level_vocab)
```

**Used By**: GPT-3, GPT-4

---

## Word Embeddings

### Static Embeddings

**Idea**: Each word has **one fixed vector** regardless of context.

### Word2Vec

**Two Architectures**:

#### 1. Skip-gram

**Objective**: Predict context words given center word.

```
The quick brown fox jumps
         ↑
    center word
    
Task: Predict {The, quick, brown, jumps} from "fox"
```

**Model**:

$$P(\text{context} | \text{center}) = \prod_{-c \leq j \leq c, j \neq 0} P(w_{t+j} | w_t)$$

Where:
$$P(o | c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in V} \exp(u_w^T v_c)}$$

- $v_c$ = center word embedding
- $u_o$ = context word embedding

**Pseudocode**:

```python
def skip_gram(center_word, context_words, vocab_size, embedding_dim):
    # Input: One-hot vector for center word
    center_onehot = to_onehot(center_word, vocab_size)
    
    # Embed center word
    center_emb = W_in @ center_onehot  # [embedding_dim]
    
    # For each context word
    loss = 0
    for context_word in context_words:
        # Compute scores
        scores = W_out @ center_emb  # [vocab_size]
        
        # Softmax
        probs = softmax(scores)
        
        # Cross-entropy loss
        context_onehot = to_onehot(context_word, vocab_size)
        loss += -log(probs @ context_onehot)
    
    return loss
```

**Negative Sampling**: Optimize with negative samples instead of full softmax.

$$\log \sigma(u_o^T v_c) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)}[\log \sigma(-u_{w_i}^T v_c)]$$

- Positive sample: Actual context word
- Negative samples: K random words (typically 5-20)

#### 2. CBOW (Continuous Bag of Words)

**Objective**: Predict center word from context.

```
The quick brown [?] jumps
                 ↑
            predict this

Task: Predict "fox" from {The, quick, brown, jumps}
```

**Model**:

$$P(\text{center} | \text{context}) = P(w_t | w_{t-c}, ..., w_{t-1}, w_{t+1}, ..., w_{t+c})$$

```python
def cbow(context_words, center_word, vocab_size, embedding_dim):
    # Embed all context words
    context_embs = [W_in @ to_onehot(w, vocab_size) for w in context_words]
    
    # Average context embeddings
    context_avg = mean(context_embs)
    
    # Predict center word
    scores = W_out @ context_avg
    probs = softmax(scores)
    
    # Loss
    center_onehot = to_onehot(center_word, vocab_size)
    loss = -log(probs @ center_onehot)
    
    return loss
```

| Aspect | Skip-gram | CBOW |
|--------|-----------|------|
| **Predicts** | Context from center | Center from context |
| **Speed** | Slower | Faster |
| **Rare words** | Better | Worse |
| **Use** | Larger datasets | Smaller datasets |

---

### GloVe (Global Vectors)

**Core Idea**: Factorize co-occurrence matrix.

**Objective**:

$$J = \sum_{i,j=1}^V f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

Where:
- $X_{ij}$ = co-occurrence count (how often word $i$ and $j$ appear together)
- $f(X_{ij})$ = weighting function (avoid overweighting frequent pairs)
- $w_i, \tilde{w}_j$ = word vectors

**Weighting Function**:

$$f(x) = \begin{cases}
(x/x_{max})^\alpha & \text{if } x < x_{max} \\
1 & \text{otherwise}
\end{cases}$$

Typically $x_{max} = 100$, $\alpha = 0.75$

**Algorithm**:

```python
def train_glove(corpus, embedding_dim, window_size):
    # Step 1: Build co-occurrence matrix
    cooccur_matrix = build_cooccurrence(corpus, window_size)
    
    # Step 2: Initialize embeddings
    W = random_init([vocab_size, embedding_dim])
    W_tilde = random_init([vocab_size, embedding_dim])
    b = zeros([vocab_size])
    b_tilde = zeros([vocab_size])
    
    # Step 3: Optimize
    for epoch in range(num_epochs):
        for (i, j), X_ij in cooccur_matrix.items():
            if X_ij == 0:
                continue
            
            # Compute loss
            diff = dot(W[i], W_tilde[j]) + b[i] + b_tilde[j] - log(X_ij)
            loss = f(X_ij) * diff**2
            
            # Gradient descent
            update_weights(W[i], W_tilde[j], b[i], b_tilde[j], loss)
    
    # Final embedding: average of W and W_tilde
    embeddings = (W + W_tilde) / 2
    return embeddings
```

---

### FastText

**Enhancement**: Use **character n-grams** in addition to words.

**Key Idea**: Word representation = sum of subword representations.

**Example**:
```
Word: "apple"

Character n-grams (n=3):
<ap, app, ppl, ple, le>

Embedding(apple) = Embedding(word_apple) 
                 + Embedding(<ap)
                 + Embedding(app)
                 + Embedding(ppl)
                 + Embedding(ple)
                 + Embedding(le>)
```

**Benefits**:
- **Morphology**: "run", "running", "runner" share subwords
- **OOV handling**: Can embed unseen words
- **Rare words**: Better representations

**Used By**: Facebook AI, multilingual embeddings

---

## Contextual Embeddings

### Problem with Static Embeddings

**Issue**: One embedding per word, ignores context.

```
"Bank of the river" vs "Bank account"
        ↓                    ↓
   [Same embedding]   [Same embedding]
```

### ELMo (Embeddings from Language Models)

**Solution**: Extract from bidirectional LSTM language model.

**Architecture**:

```
Forward LSTM  ────────────────→
  "The"  "bank" "of"  "river"

Backward LSTM ←────────────────
  "The"  "bank" "of"  "river"
```

**Embedding** = Weighted sum of all layers:

$$\text{ELMo}_k = \gamma \sum_{j=0}^L s_j h_{k,j}$$

Where:
- $h_{k,j}$ = hidden state at token $k$, layer $j$
- $s_j$ = learned weights
- $\gamma$ = scaling factor

**Usage**:
```python
def get_elmo_embedding(sentence):
    # Run bidirectional LSTM
    forward_states = forward_lstm(sentence)   # [L+1, seq_len, hidden]
    backward_states = backward_lstm(sentence) # [L+1, seq_len, hidden]
    
    # Concatenate
    states = concatenate([forward_states, backward_states], dim=-1)
    
    # Weighted sum of layers
    embeddings = sum(s_j * states[j] for j in range(L+1))
    
    return embeddings
```

---

### BERT Embeddings

**Improvement**: Transformer-based, bidirectional.

**Three Types of Embeddings**:

```
Token:      [CLS]  The   bank   is    open  [SEP]
Position:     0     1      2     3      4      5
Segment:      0     0      0     0      0      0

Final = Token Emb + Position Emb + Segment Emb
```

**Extraction**:

```python
from transformers import BertModel, BertTokenizer

def get_bert_embedding(text, layer=-2):
    """
    layer=-1: Last layer (best for classification)
    layer=-2: Second-to-last (best for semantic similarity)
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt')
    
    # Forward pass
    outputs = model(**inputs, output_hidden_states=True)
    
    # Extract layer
    embeddings = outputs.hidden_states[layer]
    
    return embeddings  # [batch, seq_len, 768]
```

**Pooling Strategies**:

1. **[CLS] token**: `embeddings[:, 0, :]`
2. **Mean pooling**: `mean(embeddings, dim=1)`
3. **Max pooling**: `max(embeddings, dim=1)`

---

## Language Modeling

### Definition

**Task**: Predict next word given previous words.

$$P(w_t | w_1, ..., w_{t-1})$$

### Types

#### 1. Causal (Autoregressive) LM

**Direction**: Left-to-right

**Example**: GPT

$$P(w_1, ..., w_n) = \prod_{i=1}^n P(w_i | w_1, ..., w_{i-1})$$

#### 2. Masked LM

**Idea**: Mask random tokens, predict them.

**Example**: BERT

```
Input:  "The [MASK] sat on the mat"
Output: "cat"
```

**Training**:
```python
def masked_lm_loss(text, mask_prob=0.15):
    tokens = tokenize(text)
    
    # Randomly mask tokens
    masked_tokens = []
    labels = []
    for token in tokens:
        if random() < mask_prob:
            masked_tokens.append("[MASK]")
            labels.append(token)
        else:
            masked_tokens.append(token)
            labels.append(-100)  # Ignore in loss
    
    # Predict masked tokens
    predictions = model(masked_tokens)
    
    # Compute loss only on masked positions
    loss = cross_entropy(predictions, labels, ignore_index=-100)
    
    return loss
```

#### 3. Permutation LM

**Idea**: Predict all tokens in random order.

**Example**: XLNet

Combines benefits of autoregressive and masked LM.

---

## Common NLP Tasks

### 1. Sequence Classification

**Task**: Classify entire sequence.

**Examples**: Sentiment analysis, spam detection

```python
def sequence_classification(text, num_classes):
    # Encode
    embeddings = encoder(text)  # [seq_len, hidden]
    
    # Pool
    pooled = mean(embeddings, dim=0)  # [hidden]
    
    # Classify
    logits = linear(pooled, num_classes)  # [num_classes]
    
    return softmax(logits)
```

### 2. Token Classification

**Task**: Classify each token.

**Examples**: NER, POS tagging

```python
def token_classification(text, num_classes):
    # Encode
    embeddings = encoder(text)  # [seq_len, hidden]
    
    # Classify each token
    logits = linear(embeddings, num_classes)  # [seq_len, num_classes]
    
    return logits
```

### 3. Sequence-to-Sequence

**Task**: Generate output sequence from input sequence.

**Examples**: Translation, summarization

```
Encoder: "Hello world" → Context vector
Decoder: Context vector → "Bonjour monde"
```

---

## Interview Insights

### Common Questions

**Q1: Explain BPE tokenization.**

**Answer**: 
1. Start with character-level vocabulary
2. Iteratively merge most frequent adjacent pairs
3. Continue until desired vocabulary size
4. Handles OOV by breaking into subwords
5. Used in GPT models

**Q2: Word2Vec skip-gram vs CBOW?**

| Aspect | Skip-gram | CBOW |
|--------|-----------|------|
| Predicts | Context from word | Word from context |
| Better for | Rare words | Frequent words |
| Speed | Slower | Faster |

**Q3: Why contextual embeddings better than static?**

**Answer**: Static embeddings (Word2Vec) assign same vector regardless of context. Contextual embeddings (ELMo, BERT) adapt based on surrounding words, capturing polysemy and syntax.

### Common Pitfalls

❌ **Confusing BPE and WordPiece**: BPE uses frequency, WordPiece uses likelihood

❌ **Not explaining negative sampling**: Critical for efficient Word2Vec training

❌ **Forgetting subword benefits**: OOV handling, morphology, rare words

---

**Next**: [AI Agents →](05-AI-AGENTS.md)


