# System Architecture

## High-Level Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                         User Interface                              │
│              (Jupyter Notebook / CLI / Web Dashboard)               │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
┌────────────────────────────────▼───────────────────────────────────┐
│                      Experiment Orchestrator                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │   Training   │  │  Evaluation  │  │Interpretability│           │
│  │   Scripts    │  │   Scripts    │  │   Scripts    │             │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
└─────────┼──────────────────┼──────────────────┼────────────────────┘
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼────────────────────┐
│                        Core Framework                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Configuration Manager                      │  │
│  │           (YAML configs + Hydra composition)                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────┐  │
│  │   Models    │  │ Environment │  │    Trustworthy AI        │  │
│  │             │  │             │  │                          │  │
│  │ • Baseline  │  │ • RAN Sim   │  │ • Uncertainty (Ensemble) │  │
│  │ • PPO-GNN   │  │ • Traffic   │  │ • Robustness (Shift)     │  │
│  │ • Ensemble  │  │ • Channels  │  │ • Interpretability(SHAP) │  │
│  └─────────────┘  └─────────────┘  └──────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Training Pipeline                          │  │
│  │  • Data Loading • Curriculum Learning • Checkpointing        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  Evaluation Framework                         │  │
│  │  • Metrics • Robustness Tests • Calibration • SHAP Analysis  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼────────────────────┐
│                         Data Storage                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │   Raw    │  │Processed │  │Synthetic │  │   Checkpoints    │   │
│  │  Traces  │  │  States  │  │ Traffic  │  │   & Logs         │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Environment (src/environment/)
- **ran_env.py**: Gymnasium-compatible RAN simulator
  - Multi-cell topology (graph structure)
  - User equipment (UE) dynamics
  - Channel models (3GPP UMa)
  - Traffic generators
  - Reward computation

### 2. Models (src/models/)

#### Encoders (src/models/encoders/)
- **gnn_encoder.py**: Graph Attention Network
  - Nodes: Base stations (cells)
  - Edges: Interference relationships
  - Features: Load, CQI, buffers
  - Output: Graph embedding

#### Policies (src/models/policies/)
- **ppo_policy.py**: Proximal Policy Optimization
  - Actor: π(a|s) - stochastic policy
  - Critic: V(s) - value function
  - Uses encoder output as state representation

#### Uncertainty (src/models/uncertainty/)
- **ensemble.py**: 5 independent policies
- **mc_dropout.py**: Monte Carlo Dropout
- **calibration.py**: ECE and reliability diagrams

### 3. Training (src/training/)
- **trainer.py**: PPO training loop
  - Rollout collection
  - Advantage estimation (GAE)
  - Policy and value updates
  - Curriculum learning

### 4. Evaluation (src/evaluation/)
- **metrics.py**: Performance metrics
  - Throughput, SLA violations, fairness
- **robustness.py**: Distribution shift tests
- **calibration.py**: Uncertainty calibration
- **interpretability.py**: SHAP analysis

## Data Flow

```
┌─────────────┐
│ Config YAML │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  1. Data Generation                     │
│     • Traffic patterns                  │
│     • Shift scenarios                   │
│     • Channel realizations              │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  2. Training                            │
│     • Sample (s,a,r,s') transitions     │
│     • Compute advantages                │
│     • Update policy                     │
│     • Log metrics                       │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  3. Checkpointing                       │
│     • Save model state                  │
│     • Save optimizer state              │
│     • Save training stats               │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  4. Evaluation                          │
│     • Load checkpoint                   │
│     • Run on test scenarios             │
│     • Compute metrics                   │
│     • Generate visualizations           │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  5. Analysis                            │
│     • Robustness under shift            │
│     • Uncertainty quantification        │
│     • Interpretability (SHAP)           │
│     • Ablation studies                  │
└─────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Modular Architecture
- Each component is independent and testable
- Easy to swap encoders (GNN ↔ Transformer)
- Easy to add new uncertainty methods

### 2. Configuration-Driven
- All hyperparameters in YAML files
- Hydra for composition and overrides
- Reproducibility through config versioning

### 3. Gymnasium Interface
- Standard RL interface for environment
- Compatible with Stable-Baselines3
- Easy integration with other RL libraries

### 4. Graph Representation
- NetworkX for cell topology
- PyTorch Geometric for GNN
- Captures spatial structure of RAN

### 5. Ensemble Uncertainty
- Multiple independent models
- Variance as uncertainty estimate
- Decision fusion for robust allocation

## Extension Points

### Adding New Models
1. Create encoder in `src/models/encoders/`
2. Implement forward() method
3. Register in config YAML
4. Use in policy network

### Adding New Metrics
1. Add method to `EvaluationMetrics` class
2. Compute during evaluation
3. Add to results dictionary
4. Visualize in notebooks

### Adding New Scenarios
1. Define in `configs/data/shift_scenarios.yaml`
2. Implement scenario logic in `ran_env.py`
3. Run evaluation with new scenario
4. Compare against baselines
