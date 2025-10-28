# Core ML Algorithms

## Table of Contents
1. [Feature Engineering](#feature-engineering)
2. [Supervised Learning](#supervised-learning)
3. [Unsupervised Learning](#unsupervised-learning)
4. [Dimensionality Reduction](#dimensionality-reduction)
5. [Model Evaluation and Tuning](#model-evaluation-and-tuning)
6. [Advanced Techniques](#advanced-techniques)
7. [Interview Insights](#interview-insights)

---

## Feature Engineering

### What is Feature Engineering?

**Definition**: Process of selecting, modifying, or creating new features from raw data to improve model performance.

**Key Principle**: Garbage in, garbage out. Model quality heavily depends on feature quality.

### Why Feature Engineering Matters

**Good features**:
- Captures relevant information
- Discriminates between classes
- Represents relationships in data

**Impact**: Can improve model accuracy by 10-50% without changing algorithm.

### Types of Feature Engineering

#### 1. Feature Selection

**Goal**: Choose most relevant features

**Methods**:
- **Filter methods**: Statistical tests (Chi-square, correlation)
- **Wrapper methods**: Try different subsets (forward/backward selection)
- **Embedded methods**: Regularization (Lasso selects features)

**Benefits**: Reduces overfitting, faster training, more interpretable

#### 2. Feature Transformation

**Scaling**: 
- Standardization: $z = \frac{x - \mu}{\sigma}$
- Normalization: $x' = \frac{x - \min}{\max - \min}$

**Encoding**:
- One-hot: Binary vectors for categorical
- Label encoding: Numeric codes
- Target encoding: Use target statistics

**Bin discretization**: Convert continuous to categorical

#### 3. Feature Creation

**Combinations**: 
- Polynomial features ($x^2, xy$)
- Interactions (product, ratio)

**Domain knowledge**:
- Time features: hour, day-of-week
- Aggregations: mean, max per group

---

## Supervised Learning

### Linear Regression

**Goal**: Predict continuous values

**Model**: $y = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$

**Intuition**: Finds line/plane that minimizes prediction error

**Cost Function**: Mean Squared Error (MSE)

$$MSE = \frac{1}{m}\sum_{i=1}^m (y_i - \hat{y}_i)^2$$

**Solution Methods**:
- Normal equation: Closed-form solution
- Gradient descent: Iterative optimization

**Assumptions**:
1. Linearity between features and target
2. Independent observations
3. Homoscedastic errors (constant variance)
4. Normal error distribution

**When to use**: Linear relationships, interpretability needed, baseline model

**Limitations**: Assumes linearity (use polynomial regression for non-linear)

### Logistic Regression

**Goal**: Binary classification (0 or 1)

**Model**: Uses sigmoid function

$$P(y=1|\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$

**Intuition**: Sigmoid outputs probability in [0,1]

**Cost Function**: Cross-entropy loss

$$\mathcal{L} = -\frac{1}{m}\sum[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$$

**Why cross-entropy**: Penalizes confident wrong predictions heavily

**Decision boundary**: Choose threshold (typically 0.5)

**Multiclass extension**: Use softmax for multiple classes

**When to use**: Binary classification, interpretable probabilities needed, baseline classifier

### Decision Trees

**How they work**: Sequentially split data based on features

**Splitting criteria**:
- **Gini impurity**: $1 - \sum p_i^2$ (measures class mixture)
- **Entropy**: $-\sum p_i \log p_i$ (information theory)
- **Information gain**: Reduction in impurity after split

**Stopping conditions**:
- Max depth
- Min samples per leaf
- Min improvement in purity

**Advantages**:
- Interpretable (if-then rules)
- Handles non-linear boundaries
- Feature importance
- No assumptions about data distribution

**Disadvantages**:
- Prone to overfitting
- Sensitive to small data changes
- Greedy (may not find optimal tree)
- Bias toward features with more levels

### Random Forest

**Concept**: Ensemble of many decision trees

**Key mechanism**: 
1. **Bagging**: Train each tree on bootstrap sample
2. **Random features**: Consider random subset at each split
3. **Aggregation**: Majority vote (classification) or average (regression)

**Why it works**:
- Reduces variance (averaging over many trees)
- Robust to noise (each tree sees different data)
- Feature randomness decorrelates trees

**Advantages over single tree**:
- Less overfitting
- Better generalization
- Estimates feature importance

**Hyperparameters**:
- Number of trees (typically 100-500)
- Max depth
- Features per split ($\sqrt{n}$ typical)

### K-Nearest Neighbors (KNN)

**How it works**: Classification/regression by majority vote of K nearest points

**Distance metrics**:
- **Euclidean**: $||\mathbf{x}_i - \mathbf{x}_j||_2$
- **Manhattan**: $||\mathbf{x}_i - \mathbf{x}_j||_1$
- **Hamming**: For categorical features

**Choose K**:
- Small K (K=1): Low bias, high variance (sensitive to noise)
- Large K: High bias, low variance (smoother boundaries)

**Typical K**: Odd numbers (3, 5, 7, 11)

**Advantages**:
- Simple, no training phase
- Naturally handles multi-class
- Non-parametric (no distribution assumptions)

**Disadvantages**:
- Expensive prediction (compare to all training points)
- Sensitive to irrelevant features
- Curse of dimensionality (high-d space, neighbors become far)

**When to use**: 
- Non-linear boundaries
- Small datasets
- Multi-class problems

### Support Vector Machine (SVM)

**Key idea**: Find hyperplane that maximizes margin between classes

**Mathematical formulation**:
- **Hard margin**: Classify all points correctly
- **Soft margin**: Allow some misclassification (C parameter)

**Support vectors**: Only points on margin matter

**Kernels** (handle non-linear):
- **Linear**: $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T\mathbf{x}_j$
- **Polynomial**: $(\mathbf{x}_i^T\mathbf{x}_j + 1)^d$
- **RBF**: $\exp(-\gamma||\mathbf{x}_i - \mathbf{x}_j||^2)$

**Advantages**:
- Effective in high dimensions
- Memory efficient (uses support vectors only)
- Versatile (non-linear via kernels)
- Regularization built-in

**Disadvantages**:
- Doesn't work well on large datasets
- Requires feature scaling
- Difficult to interpret
- Kernel choice is critical

**When to use**: High-dimensional data, clear margin of separation, binary classification

### Naive Bayes

**Key idea**: Use Bayes theorem with independence assumption

$$P(y|\mathbf{x}) = \frac{P(\mathbf{x}|y)P(y)}{P(\mathbf{x})}$$

**Naive assumption**: Features independent given class

$$P(\mathbf{x}|y) = \prod_{i=1}^n P(x_i|y)$$

**Variants**:
- **Gaussian NB**: Continuous features (assumes normal distribution)
- **Multinomial NB**: Count data (word counts)
- **Bernoulli NB**: Binary features

**Advantages**:
- Fast training and prediction
- Works well with small data
- Handles multi-class naturally
- Works with high dimensions

**Disadvantages**:
- Independence assumption rarely true
- Requires all features (missing data problematic)
- Sensitive to data scarcity

**When to use**: Text classification, baseline classifier, small datasets

---

## Unsupervised Learning

### What is Unsupervised Learning?

**Definition**: Learning patterns from unlabeled data

**Tasks**:
- Clustering: Group similar data
- Dimensionality reduction: Reduce feature count
- Anomaly detection: Find outliers

**Use cases**: 
- Discover hidden patterns
- Preprocessing for supervised learning
- Data exploration

### K-Means Clustering

**How it works**:
1. Initialize K cluster centroids randomly
2. Assign each point to nearest centroid
3. Update centroids as mean of assigned points
4. Repeat until convergence

**Distance**: Typically Euclidean

**Convergence**: When assignments don't change

**K selection**:
- **Elbow method**: Plot cost vs K, choose elbow
- **Domain knowledge**: Natural groupings
- **Silhouette analysis**: Measure cluster quality

**Advantages**:
- Fast and scalable
- Simple to implement
- Works well with spherical clusters

**Disadvantages**:
- Need to specify K
- Sensitive to initialization (local optima)
- Assumes spherical clusters
- Sensitive to outliers

**Applications**: Customer segmentation, image compression, anomaly detection

### Hierarchical Clustering

**Types**:
- **Agglomerative** (bottom-up): Start with each point as cluster, merge closest
- **Divisive** (top-down): Start with all points, split recursively

**Linkage criteria**:
- **Single**: Minimum distance between clusters
- **Complete**: Maximum distance
- **Average**: Mean distance
- **Ward**: Minimizes within-cluster variance

**Dendrogram**: Tree visualization showing cluster hierarchy

**Advantages**:
- No need to specify number of clusters
- Intuitive visualization
- Deterministic (same result each time)

**Disadvantages**:
- Computationally expensive ($O(n^2 \log n)$)
- Sensitive to outliers
- Can't undo merges

**When to use**: Don't know K, want hierarchy, small-medium datasets

### DBSCAN Clustering

**Key concept**: Density-based clustering

**Parameters**:
- **eps ($\epsilon$)**: Maximum distance to consider neighbors
- **min_samples**: Minimum points to form cluster

**Types of points**:
- **Core**: Has $\geq$ min_samples neighbors within eps
- **Border**: Neighbor of core point
- **Noise**: Neither core nor border

**How it works**:
1. Pick random unvisited point
2. If has enough neighbors, start cluster
3. Expand cluster to neighbors
4. Mark as visited
5. Repeat

**Advantages**:
- Finds clusters of arbitrary shape
- Identifies noise/outliers
- Doesn't require specifying K
- Robust to outliers

**Disadvantages**:
- Sensitive to eps and min_samples parameters
- Struggles with varying density
- Border points may belong to multiple clusters

**When to use**: Unknown cluster shapes, noise present, varying densities acceptable

---

## Dimensionality Reduction

### Why Reduce Dimensions?

**Reasons**:
1. **Curse of dimensionality**: Performance degrades in high dimensions
2. **Visualization**: Can only plot 2-3 dimensions
3. **Collinearity**: Remove redundant features
4. **Noise reduction**: Main signal in fewer dimensions
5. **Storage/compute**: Smaller models faster

### Principal Component Analysis (PCA)

**Goal**: Project data onto directions of maximum variance

**Mathematical approach**:
1. Center data (subtract mean)
2. Compute covariance matrix
3. Find eigenvectors (principal components)
4. Project onto top K components

**Number of components**: Choose K such that captures X% variance (typically 95%)

**Intuition**: 
- PC1 captures most variance
- PC2 captures next most (orthogonal to PC1)
- And so on...

**Use cases**:
- Visualize high-d data
- Remove redundancy
- Noise reduction
- Before other ML algorithms

**Limitations**:
- Assumes linear relationships
- Not interpretable (linear combinations)
- Sensitive to scaling

### Feature Selection Techniques

**Filter Methods**:
- **Correlation**: Remove highly correlated features
- **Variance**: Remove low variance
- **Chi-square**: Independence test for categorical

**Wrapper Methods**:
- **Forward selection**: Start empty, add best feature each step
- **Backward elimination**: Start with all, remove worst each step

**Embedded Methods**:
- **Lasso**: L1 regularization automatically selects features
- **Tree-based**: Feature importance from trees

**RFE (Recursive Feature Elimination)**: Remove least important features iteratively

---

## Model Evaluation and Tuning

### Evaluation Metrics

#### Classification Metrics

**Confusion Matrix**:
```
                 Predicted
             Negative    Positive
Actual  Neg     TN         FP
        Pos     FN         TP
```

**Accuracy**: $\frac{TP+TN}{TP+TN+FP+FN}$ (overall correctness)

**Precision**: $\frac{TP}{TP+FP}$ (of positive predictions, how many correct)

**Recall**: $\frac{TP}{TP+FN}$ (of actual positives, how many found)

**F1-Score**: $2 \frac{Precision \times Recall}{Precision + Recall}$ (harmonic mean)

**ROC-AUC**: Area under ROC curve (TPR vs FPR) - works for probability outputs

#### Regression Metrics

**MSE**: $\frac{1}{m}\sum(y - \hat{y})^2$ (penalizes large errors)

**RMSE**: $\sqrt{MSE}$ (same units as target)

**MAE**: $\frac{1}{m}\sum|y - \hat{y}|$ (robust to outliers)

**R²**: $1 - \frac{\sum(y-\hat{y})^2}{\sum(y-\bar{y})^2}$ (explained variance, 1 is perfect)

### Cross Validation

**Purpose**: Unbiased estimate of model performance

**K-Fold CV**:
- Split data into K folds
- Train on K-1, test on remaining fold
- Repeat K times
- Average performance

**Typical K**: 5 or 10

**Stratified K-Fold**: For classification, maintains class distribution

**Advantages**: 
- Uses all data for training and testing
- Reduces variance in performance estimate

### Regularization

**L1 (Lasso)**: $\lambda\sum|\theta_i|$
- Encourages sparsity (many weights → 0)
- Automatic feature selection
- May remove highly correlated features arbitrarily

**L2 (Ridge)**: $\lambda\sum\theta_i^2$
- Keeps weights small
- Reduces overfitting
- Doesn't drop features

**Elastic Net**: Combines L1 + L2 ($\alpha\lambda\sum|\theta_i| + (1-\alpha)\lambda\sum\theta_i^2$)

### Hyperparameter Tuning

**Manual tuning**: Try different values based on experience

**Grid search**: Try all combinations of hyperparameters (exhaustive)

**Random search**: Sample randomly from hyperparameter space (often better)

**Bayesian optimization**: Learn best hyperparameters efficiently

### Bias-Variance Tradeoff

**Bias**: Error from oversimplifying (underfitting)

**Variance**: Error from sensitivity to training set (overfitting)

**Tradeoff**: Can't minimize both simultaneously

**Underfitting**: High bias, low variance
- Simple model
- Can't capture patterns
- Both training and test error high

**Overfitting**: Low bias, high variance
- Complex model
- Fits training noise
- Training error low, test error high

**Goal**: Balance (sweet spot)

**Solutions**:
- **High bias**: Increase model complexity
- **High variance**: More data, regularization, ensemble

---

## Advanced Techniques

### Reinforcement Learning

**Definition**: Agent learns by interacting with environment

**Components**:
- **Agent**: Learning system
- **Environment**: External world
- **State**: Current situation
- **Action**: What agent can do
- **Reward**: Signal for good/bad actions
- **Policy**: How agent chooses actions

**Key idea**: Maximize cumulative reward (long-term)

**Q-Learning**: Learn action-value function

$Q(s,a)$ = expected future reward from state $s$ taking action $a$

**Applications**: Game playing, robotics, recommendation systems

### Semi-Supervised Learning

**Definition**: Uses labeled + unlabeled data

**Why**: Labeling is expensive

**Assumptions**:
- **Smoothness**: Similar inputs → similar outputs
- **Cluster**: Points in same cluster → same class
- **Manifold**: Data lies on low-dimensional manifold

**Approaches**:
- **Self-training**: Train on labeled, predict unlabeled, add high-confidence to training set
- **Co-training**: Multiple views of data
- **Generative models**: Model $P(x|y)$ and $P(y)$, predict $P(y|x)$ for unlabeled

### Self-Supervised Learning

**Definition**: Create supervisory signal from data itself

**Examples**:
- **Image**: Rotate image, predict rotation angle
- **Text**: Mask tokens, predict masked tokens (BERT)
- **Video**: Predict future frame

**Advantage**: Use vast amounts of unlabeled data

**Use**: Pretraining before fine-tuning on downstream task

### Ensemble Learning

**Core idea**: Combine multiple models for better performance

**Types**:

**1. Bagging (Bootstrap Aggregating)**:
- Train different models on different data samples
- Average predictions
- Reduces variance
- Example: Random Forest

**2. Boosting**:
- Train models sequentially, each focuses on previous errors
- Weighted combination
- Reduces bias
- Example: AdaBoost, Gradient Boosting, XGBoost

**3. Stacking**:
- Train base models
- Train meta-model on base model predictions
- Learns how to best combine

**Why ensembles work**: 
- Diversity in models → compensate for individual errors
- "Wisdom of crowds"
- Decreases variance (bagging) or bias (boosting)

---

## Interview Insights

### Common Questions

**Q1: When would you use SVM vs neural network?**

**Answer**:

| Aspect | SVM | Neural Network |
|--------|-----|----------------|
| **Data size** | Small (<10K) | Large |
| **Feature count** | High dimensions OK | Many parameters |
| **Interpretability** | Medium | Low |
| **Non-linearity** | Kernel methods | Universal approximator |
| **Data requirements** | Less | More |

Choose SVM for small, high-dimensional, clear margins. NN for large datasets, complex patterns.

**Q2: Explain bias-variance tradeoff.**

**Answer**:
- **Bias**: Model too simple, misses patterns
- **Variance**: Model too complex, sensitive to training data
- **Tradeoff**: Can't have both low bias AND low variance
- **Goal**: Sweet spot (generalization)
- **Solutions**: Regularization, ensemble methods, more data

**Q3: How choose between algorithms?**

**Answer**: Consider:
1. **Data size**: Small → KNN/non-parametric, Large → NN
2. **Interpretability**: Tree models vs black box
3. **Linearity**: Linear → Linear/Logistic, Non-linear → Tree/NN
4. **Features**: High-d → SVM, Low-d → Most algorithms work
5. **Training time**: Real-time → Simple models
6. **Prediction time**: Fast inference needed → Simpler models

**Q4: What's difference between K-means and hierarchical clustering?**

**Answer**:
- **K-means**: Need to specify K, fast, spherical clusters, single result
- **Hierarchical**: Don't need K, creates dendrogram, any shape, multiple results
- **When hierarchical**: Unknown K, want multiple solutions, interpretable hierarchy
- **When K-means**: Known K, large dataset, fast needed

**Q5: How does PCA reduce dimensions?**

**Answer**:
1. Find directions of maximum variance (principal components)
2. Project data onto fewer components
3. Keep components explaining most variance
4. Result: Lower dimension with minimal information loss
5. **Geometric**: Rotate axes to align with data spread

### Common Pitfalls

❌ **Using wrong metric**: Accuracy on imbalanced data

❌ **Data leakage**: Future info in training

❌ **Overfitting**: Perfect training, poor test

❌ **Feature mismatch**: Train/test distributions differ

❌ **Ignoring assumptions**: E.g., linear regression needs linear relationship

---

**Next**: [ML Mathematics →](10-ML-MATHEMATICS.md)
