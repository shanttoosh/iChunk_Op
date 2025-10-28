# ML Mathematics

## Table of Contents
1. [Linear Algebra](#linear-algebra)
2. [Calculus](#calculus)
3. [Probability & Statistics](#probability--statistics)
4. [Optimization](#optimization)
5. [Loss Functions](#loss-functions)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Interview Insights](#interview-insights)

---

## Linear Algebra

### Vectors

**Definition**: Ordered list of numbers, $\mathbf{v} \in \mathbb{R}^n$

**Operations**:
- **Dot Product**: $\mathbf{v}^T\mathbf{w} = \sum_{i=1}^n v_i w_i = ||\mathbf{v}|| ||\mathbf{w}|| \cos\theta$
- **Norm**: $||\mathbf{v}||_2 = \sqrt{\sum v_i^2}$

### Matrices

**Multiplication**:

$\mathbf{C} = \mathbf{AB}$ where $C_{ij} = \sum_k A_{ik} B_{kj}$

**Transpose**: $(\mathbf{A}^T)_{ij} = A_{ji}$

**Properties**:
- $(\mathbf{AB})^T = \mathbf{B}^T\mathbf{A}^T$
- Not commutative: $\mathbf{AB} \neq \mathbf{BA}$

### Eigenvalues & Eigenvectors

$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$

$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$

**Eigendecomposition**: $\mathbf{A} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^{-1}$

**Use**: PCA, dimensionality reduction

### SVD (Singular Value Decomposition)

$\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$

Where:
- $\mathbf{U}$: left singular vectors
- $\mathbf{\Sigma}$: singular values (diagonal)
- $\mathbf{V}$: right singular vectors

**Use**: Low-rank approximation, noise reduction

$$\mathbf{A}_k = \mathbf{U}_k\mathbf{\Sigma}_k\mathbf{V}_k^T$$ (rank-k approximation)

---

## Calculus

### Derivatives

**Definition**: $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$

**Chain Rule**: $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$

**Partial**: $\frac{\partial f}{\partial x_i} = \frac{\partial}{\partial x_i} f(x_1, ..., x_n)$

### Gradients

**Gradient Vector**: $\nabla f = \left[\frac{\partial f}{\partial x_1}, ..., \frac{\partial f}{\partial x_n}\right]^T$

**Gradient Descent**: $\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$

### Common Derivatives

$\frac{d}{dx}x^n = nx^{n-1}$

$\frac{d}{dx}e^x = e^x$

$\frac{d}{dx}\log(x) = \frac{1}{x}$

$\frac{d}{dx}\sigma(x) = \sigma(x)(1-\sigma(x))$ where $\sigma$ = sigmoid

$\frac{d}{dx}\text{ReLU}(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$

---

## Probability & Statistics

### Probability Basics

**Bayes Theorem**:

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

**Expectation**: $\mathbb{E}[X] = \sum x P(x)$

**Variance**: $\text{Var}(X) = \mathbb{E}[(X-\mu)^2] = \mathbb{E}[X^2] - \mu^2$

### Distributions

**Gaussian**: $P(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$

**Bernoulli**: $P(x=1) = p$, $P(x=0) = 1-p$

**Multinomial**: $P(\mathbf{x}) = \frac{n!}{x_1!...x_k!} p_1^{x_1}...p_k^{x_k}$

### Maximum Likelihood Estimation

**Likelihood**: $\mathcal{L}(\theta) = \prod_{i=1}^n P(x_i|\theta)$

**Log-Likelihood**: $\log\mathcal{L}(\theta) = \sum_{i=1}^n \log P(x_i|\theta)$

**MLE**: $\hat{\theta} = \arg\max_\theta \log\mathcal{L}(\theta)$

### Hypothesis Testing

**t-test**: $t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}$

**p-value**: Probability of observing data given null hypothesis

**Significance Level**: $\alpha = 0.05$ (reject if p < $\alpha$)

---

## Optimization

### Convex Optimization

**Convex function**: $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$

**Local minimum = global minimum**

### Gradient Descent Variants

**SGD**: $\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}_i(\theta_t)$

**Momentum**: $v_t = \beta v_{t-1} + (1-\beta)\nabla\mathcal{L}$, $\theta_t = \theta_{t-1} - \eta v_t$

**Adam**: See [Neural Networks guide](06-NEURAL-NETWORKS.md)

### Learning Rate Schedule

**Step decay**: $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$

**Exponential**: $\eta_t = \eta_0 \gamma^t$

**Cosine**: $\eta_t = \eta_{\min} + (\eta_{\max} - \eta_{\min}) \frac{1+\cos(\pi t/T)}{2}$

---

## Loss Functions

### Regression Losses

**MSE**: $\mathcal{L} = \frac{1}{m}\sum(y - \hat{y})^2$

**MAE**: $\mathcal{L} = \frac{1}{m}\sum|y - \hat{y}|$

**Huber**: $\mathcal{L} = \begin{cases} \frac{1}{2}(y-\hat{y})^2 & \text{if } |y-\hat{y}| \leq \delta \\ \delta|y-\hat{y}| - \frac{\delta^2}{2} & \text{otherwise} \end{cases}$

### Classification Losses

**Cross-Entropy**: $\mathcal{L} = -\sum y \log(\hat{y})$

**Hinge**: $\mathcal{L} = \max(0, 1 - y\hat{y})$ (SVM)

### Regularization

**L1 (Lasso)**: $\lambda \sum|\theta_i|$

**L2 (Ridge)**: $\lambda \sum\theta_i^2$

**Elastic Net**: $\lambda_1 \sum|\theta_i| + \lambda_2 \sum\theta_i^2$

---

## Evaluation Metrics

### Classification

**Confusion Matrix**:

```
                 Predicted
             Negative    Positive
Actual  Neg     TN         FP
        Pos     FN         TP
```

**Accuracy**: $\frac{TP+TN}{TP+TN+FP+FN}$

**Precision**: $\frac{TP}{TP+FP}$

**Recall**: $\frac{TP}{TP+FN}$

**F1**: $2 \frac{P \cdot R}{P + R}$

**ROC-AUC**: Area under ROC curve (TPR vs FPR)

### Regression

**RMSE**: $\sqrt{\frac{1}{m}\sum(y-\hat{y})^2}$

**MAE**: $\frac{1}{m}\sum|y-\hat{y}|$

**R²**: $1 - \frac{\sum(y-\hat{y})^2}{\sum(y-\bar{y})^2}$

---

## Interview Insights

### Common Questions

**Q1: Explain gradient descent.**

**Answer**: Iteratively move in direction of negative gradient to minimize loss. Learning rate controls step size. Converges to local minimum for convex functions.

**Q2: What is cross-entropy loss?**

**Answer**: Measures difference between predicted and true distributions. Encourages confident, correct predictions. Always non-negative, 0 when perfect match.

**Q3: Derive backpropagation.**

**Answer**: Apply chain rule backwards through network. Each layer's gradient depends on next layer's gradient. See [Neural Networks guide](06-NEURAL-NETWORKS.md).

---

**Next**: [System Design →](11-SYSTEM-DESIGN.md)


