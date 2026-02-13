# Reproducibility Checklist

## Environment Setup
- [ ] Python 3.9.13 installed
- [ ] All dependencies from requirements.txt installed with exact versions
- [ ] CUDA 11.8 configured (if using GPU)
- [ ] Random seeds set in all scripts (seed=42)

## Data Generation
- [ ] Traffic generation script executed with fixed seed
- [ ] All shift scenarios generated
- [ ] Data checksums verified (see data/checksums.txt)
- [ ] Train/val/test split is 70/15/15

## Training
- [ ] Deterministic mode enabled in PyTorch
- [ ] All hyperparameters logged
- [ ] Training curves saved to TensorBoard
- [ ] Model checkpoints include full configuration

## Evaluation
- [ ] Test set never used during training
- [ ] Evaluation seeds fixed
- [ ] Statistical significance tests performed (p < 0.05)
- [ ] Results averaged over 5 random seeds with confidence intervals

## Code Quality
- [ ] Type hints added to all functions
- [ ] Docstrings in NumPy style
- [ ] Unit tests pass (pytest)
- [ ] Code formatted with black

## Documentation
- [ ] README contains installation instructions
- [ ] Technical report in docs/technical_report.pdf
- [ ] Demo notebook runs end-to-end
- [ ] All results reproducible

## Hardware Requirements
- GPU: NVIDIA RTX 3090/4090 (24GB VRAM) or equivalent
- CPU: 8+ cores for parallel data generation
- RAM: 32GB minimum
- Storage: 50GB for data and checkpoints

## Time Requirements
- Data generation: ~2 hours
- Baseline training: ~1 hour
- GNN-PPO training: ~12 hours per model
- Ensemble (5 models): ~60 hours
- Full evaluation: ~4 hours

Total: ~79 hours (can parallelize ensemble training)
