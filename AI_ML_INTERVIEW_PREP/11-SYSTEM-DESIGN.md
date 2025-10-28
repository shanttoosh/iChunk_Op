# ML System Design

## Table of Contents
1. [ML Pipeline Architecture](#ml-pipeline-architecture)
2. [Model Training](#model-training)
3. [Model Serving](#model-serving)
4. [Monitoring & Observability](#monitoring--observability)
5. [Scalability](#scalability)
6. [A/B Testing](#ab-testing)
7. [Interview Insights](#interview-insights)

---

## ML Pipeline Architecture

### Complete ML System

```
Data Sources
    ↓
┌─────────────┐
│  Data      │
│ Ingestion  │
└─────────────┘
    ↓
┌─────────────┐
│   Feature   │
│  Store      │
└─────────────┘
    ↓
┌─────────────┐
│  Training   │
│  Pipeline   │
└─────────────┘
    ↓
┌─────────────┐
│   Model     │
│  Registry   │
└─────────────┘
    ↓
┌─────────────┐
│  Model      │
│  Serving    │
└─────────────┘
    ↓
  End Users
```

### Components

1. **Data Ingestion**: Kafka, Flink, Kinesis
2. **Feature Store**: Feast, Tecton
3. **Training Pipeline**: Airflow, Kubeflow
4. **Model Registry**: MLflow, Weights & Biases
5. **Model Serving**: TensorFlow Serving, TorchServe, KServe

---

## Model Training

### Distributed Training

**Data Parallelism**:
- Split data across workers
- Each computes gradients
- Average gradients

**Model Parallelism**:
- Split model across devices
- For very large models

```python
# PyTorch DistributedDataParallel
import torch.distributed as dist

model = DDP(model)

for batch in dataloader:
    outputs = model(batch)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

### Experiment Tracking

**Track**:
- Hyperparameters
- Metrics
- Artifacts (models, data)
- Code versions

**Tools**: MLflow, Weights & Biases

### CI/CD for ML

**Pipeline**:
```
Code Change → Unit Tests → Train → Validate → Deploy
                                      ↓
                                  Reject if metrics below threshold
```

---

## Model Serving

### Deployment Patterns

**1. Batch Prediction**:
- Run periodically
- Large datasets
- Cost-effective

**2. Real-time Inference**:
- API-based
- Low latency (<100ms)
- More expensive

### Serving Infrastructure

```
Client Request
    ↓
Load Balancer
    ↓
┌──────────────┬──────────────┬──────────────┐
│  Worker 1    │  Worker 2    │  Worker N    │
│ (GPU/CPU)    │ (GPU/CPU)    │ (GPU/CPU)    │
└──────────────┴──────────────┴──────────────┘
    ↓
Model Cache (Redis/Memcached)
```

### Optimization

**1. Batching**:
- Process multiple requests together
- Amortize fixed overhead

**2. Model Optimization**:
- Quantization (FP32 → INT8)
- Pruning (remove redundant weights)
- Knowledge distillation

```python
# TensorRT for inference optimization
import tensorrt as trt

builder = trt.Builder(logger)
network = builder.create_network()

# Parse ONNX model
parser = trt.OnnxParser(network, logger)
parser.parse_from_file(model_path)

# Build engine
config = builder.create_builder_config()
engine = builder.build_engine(network, config)
```

---

## Monitoring & Observability

### Metrics to Track

**1. Model Performance**:
- Accuracy/Precision/Recall
- Prediction latency
- Throughput

**2. Data Drift**:
- Feature distributions change
- Affects model performance

**3. System Health**:
- GPU/CPU utilization
- Memory usage
- Error rates

### Alerting

```python
def monitor_model(model_id):
    # Get predictions
    recent_preds = get_recent_predictions(model_id)
    
    # Check latency
    if recent_preds.latency > threshold:
        alert("High latency detected")
    
    # Check accuracy
    if recent_preds.accuracy < baseline - margin:
        alert("Performance degradation")
    
    # Check data drift
    if has_distribution_shift(recent_preds.features):
        alert("Data drift detected")
```

### Observability Tools

- **Prometheus**: Metrics collection
- **Grafana**: Dashboards
- **Evidently AI**: Data drift detection

---

## Scalability

### Horizontal Scaling

**Add more workers**:
```
1 Worker  → 10 QPS
10 Workers → 100 QPS
```

**Challenges**:
- Load balancing
- State management
- Consistency

### Vertical Scaling

**Upgrade hardware**:
- Better GPU
- More memory

**Limits**: Single machine constraints

### Caching

**Cache** predictions for common inputs:
```python
import redis

cache = redis.Redis()

def predict_with_cache(query):
    # Check cache
    cached = cache.get(query)
    if cached:
        return cached
    
    # Compute
    prediction = model.predict(query)
    
    # Cache
    cache.setex(query, ttl=3600, value=prediction)
    
    return prediction
```

---

## A/B Testing

### Framework

**Setup**:
1. Split traffic (50/50 or 90/10)
2. Deploy A (baseline) and B (new model)
3. Collect metrics
4. Statistical significance test
5. Decide: Keep A, switch to B, or continue

**Statistical Test**:

$$z = \frac{\hat{p}_A - \hat{p}_B}{\sqrt{\frac{\hat{p}(1-\hat{p})}{n_A} + \frac{\hat{p}(1-\hat{p})}{n_B}}}$$

Where $\hat{p}$ = pooled proportion

### Metrics

- Click-through rate
- Conversion rate
- Revenue per user
- Engagement metrics

```python
def ab_test(control_metric, treatment_metric, alpha=0.05):
    # Compute z-score
    z = compute_z_score(control_metric, treatment_metric)
    
    # Two-tailed test
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    if p_value < alpha:
        return "Significant difference, reject null hypothesis"
    else:
        return "No significant difference"
```

---

## Interview Insights

### Common Questions

**Q1: Design ML system for 1M users.**

**Answer**:
1. **Data**: Collect user interactions (features: clicks, views, time)
2. **Training**: Weekly retraining on new data
3. **Serving**: Load balanced, 10+ workers, caching
4. **Monitoring**: Latency p95, accuracy tracking
5. **Scaling**: Auto-scale workers based on load

**Q2: How handle model drift?**

**Answer**:
1. **Monitor**: Track prediction distributions over time
2. **Detect**: Statistical tests (KS-test, PSI)
3. **Alert**: When drift exceeds threshold
4. **Retrain**: Update model on recent data
5. **Rollback**: Keep old model for safety

**Q3: How to roll out new model safely?**

**Answer**:
1. **Shadow mode**: Run in parallel, log predictions, don't affect users
2. **Canary**: 5% traffic → monitor
3. **Gradual**: 10%, 50%, 100% if metrics stable
4. **Rollback plan**: Quick revert if issues

### Common Pitfalls

❌ **Not monitoring**: Model degrades silently

❌ **Single point of failure**: Need redundancy

❌ **No rollback**: Can't recover from bad deployment

❌ **Ignoring costs**: LLM inference expensive

---

**Next**: [Interview Questions →](12-INTERVIEW-QUESTIONS.md)


