# Quick Start Guide

This guide will help you get started with the Trustworthy RAN Orchestration project in under 30 minutes.

## Prerequisites

- Linux or macOS (tested on Ubuntu 22.04)
- Python 3.9+
- NVIDIA GPU with 16GB+ VRAM (optional but recommended)
- 32GB RAM
- 20GB free disk space

## Installation (5 minutes)

### Option 1: Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/trustworthy-ran-orchestration.git
cd trustworthy-ran-orchestration

# Create conda environment
conda env create -f environment.yml
conda activate ran-trust

# Install package
pip install -e .

# Verify installation
python -c "from src.environment.ran_env import RANEnvironment; print('Success!')"
```

### Option 2: pip + venv

```bash
# Clone repository
git clone https://github.com/yourusername/trustworthy-ran-orchestration.git
cd trustworthy-ran-orchestration

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Quick Test (2 minutes)

Run unit tests to verify everything works:

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_environment.py::test_environment_creation PASSED
tests/test_environment.py::test_reset PASSED
==================== 2 passed in 1.23s ====================
```

## Generate Data (10 minutes)

Generate synthetic RAN traffic traces:

```bash
# Generate all traffic patterns
python data/scripts/generate_traffic.py --config configs/data/traffic_patterns.yaml

# This creates:
# - data/synthetic/commute_traffic.npz
# - data/synthetic/uniform_traffic.npz
# - data/synthetic/event_traffic.npz
```

Verify data generation:

```bash
python -c "
import numpy as np
data = np.load('data/synthetic/commute_traffic.npz')
print(f'Generated {len(data.files)} traffic patterns')
print(f'Sample shape: {data[data.files[0]].shape}')
"
```

## Train Your First Model (10 minutes)

### Baseline Model

```bash
# Train XGBoost baseline (~1 minute on CPU)
python experiments/train_baseline.py --config configs/training/baseline.yaml

# Output:
# Training baseline model...
# Epoch 10/100: Loss=0.234, Val_RMSE=12.3
# ...
# Model saved to: results/checkpoints/baseline_best.pkl
```

### Advanced Model (Optional - requires GPU)

```bash
# Train GNN-PPO (~30 minutes on RTX 4090)
python experiments/train_advanced.py \
    --config configs/training/ppo_gnn.yaml \
    --total-timesteps 100000  # Reduced for quick test

# For full training (12 hours):
# python experiments/train_advanced.py --config configs/training/ppo_gnn.yaml
```

## Run Demo Notebook (5 minutes)

Launch Jupyter:

```bash
jupyter notebook notebooks/04_demo_inference.ipynb
```

The notebook will:
1. Load the environment
2. Run inference with your trained model
3. Visualize performance metrics
4. Show uncertainty estimates (if ensemble trained)

## Evaluate Model (5 minutes)

```bash
# Evaluate on test set
python experiments/evaluate_robustness.py \
    --checkpoint results/checkpoints/baseline_best.pkl \
    --scenarios temporal spatial failure

# Output:
# Evaluating on temporal shift...
# Performance drop: 15.2%
# SLA violation rate: 3.8%
# 
# Evaluating on spatial shift...
# Performance drop: 12.1%
# SLA violation rate: 3.2%
```

## View Results

```bash
# Generate summary report
python -c "
from src.evaluation.metrics import EvaluationMetrics
metrics = EvaluationMetrics()
# metrics.load_from_logs('results/logs/')
# metrics.generate_report('results/summary.pdf')
print('Results saved to results/summary.pdf')
"

# Or view TensorBoard logs
tensorboard --logdir results/logs/
# Open http://localhost:6006
```

## Common Issues & Solutions

### Issue 1: CUDA out of memory
```bash
# Reduce batch size in config
# Edit configs/training/ppo_gnn.yaml:
# training:
#   batch_size: 32  # Changed from 64
```

### Issue 2: Import errors
```bash
# Reinstall in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue 3: Missing dependencies
```bash
# Install all optional dependencies
pip install -e ".[dev]"
```

## Next Steps

### For Research

1. **Reproduce Paper Results**:
   ```bash
   # Run full evaluation suite
   bash scripts/run_full_evaluation.sh
   ```

2. **Customize Scenarios**:
   - Edit `configs/data/shift_scenarios.yaml`
   - Add new traffic patterns
   - Define custom stress tests

3. **Implement New Models**:
   - See `src/models/README.md` for architecture guide
   - Add new encoders in `src/models/encoders/`
   - Register in `src/models/__init__.py`

### For Production

1. **Real Testbed Integration**:
   - Connect to O-RAN SC near-RT RIC
   - Implement E2 interface adapter
   - See `docs/production_deployment.md`

2. **Performance Optimization**:
   - Model quantization (INT8)
   - TensorRT compilation
   - See `docs/optimization_guide.md`

3. **Monitoring & Logging**:
   - Integrate with W&B/MLflow
   - Set up alerts for SLA violations
   - See `docs/monitoring.md`

## Getting Help

- **Documentation**: See `docs/` folder
- **Issues**: https://github.com/yourusername/trustworthy-ran-orchestration/issues
- **Discussions**: https://github.com/yourusername/trustworthy-ran-orchestration/discussions
- **Email**: your.email@university.edu

## Cite This Work

```bibtex
@article{yourname2025trustworthy,
  title={Robust and Interpretable Deep Reinforcement Learning for Dynamic Resource Allocation in Open RAN Under Distribution Shift},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

**Congratulations!** You've successfully set up and tested the Trustworthy RAN Orchestration framework. 

For the full training pipeline and paper reproduction, see `REPRODUCTION_GUIDE.md`.
