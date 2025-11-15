# LLM Configuration Guide

Complete guide to configuring language models for the proof sketcher pipeline with fine-grained control over model selection at different stages.

---

## Table of Contents

1. [Overview](#overview)
2. [4-Tier Model Architecture](#4-tier-model-architecture)
3. [Quick Start](#quick-start)
4. [Configuration Methods](#configuration-methods)
5. [YAML Configuration](#yaml-configuration)
6. [Python API](#python-api)
7. [Hydra Integration](#hydra-integration)
8. [Model Tier Recommendations](#model-tier-recommendations)
9. [Cost & Quality Tradeoffs](#cost--quality-tradeoffs)
10. [Advanced Usage](#advanced-usage)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The proof sketcher pipeline supports **configurable LLM selection** at four distinct stages:

1. **Perspective 1**: Primary strategist + Gemini reviewer
2. **Perspective 2**: Secondary strategist + Codex reviewer
3. **Synthesis**: All synthesis/consensus agents
4. **Fast**: Simple extraction and formatting tasks

This enables cost optimization by using expensive models only where needed, while maintaining quality with cheaper models for simple tasks.

**Key Features:**
- **YAML-based configuration** with Hydra support
- **4 preset configurations**: default, cost_optimized, quality_optimized, development
- **Fine-grained control**: Configure each model tier independently
- **Factory functions**: Convenient agent creation from config
- **Type-safe**: OmegaConf-compatible dataclasses with validation

---

## 4-Tier Model Architecture

### Tier 1: Perspective 1 (Primary Proof Perspective)

**Model**: `perspective_1_model`

**Components:**
- **SketchStrategist (primary)**: Classical techniques focus (lemmas, theorems, standard methods)
- **Gemini SketchRefereeAgent**: Primary validation review pipeline
  - CompletenessCorrectnessAgent
  - LogicalFlowValidation
  - DependencyValidation
  - TechnicalDeepDiveValidation

**Reasoning Complexity**: HIGH - Requires deep mathematical reasoning and proof strategy generation

**Recommended Models**: GPT-4 Turbo, Claude Opus, or other strong reasoning models

---

### Tier 2: Perspective 2 (Secondary Proof Perspective)

**Model**: `perspective_2_model`

**Components:**
- **SketchStrategist (secondary)**: Fragile Gas theory focus (framework-specific methods)
- **Codex SketchRefereeAgent**: Secondary validation review pipeline
  - CompletenessCorrectnessAgent
  - LogicalFlowValidation
  - DependencyValidation
  - TechnicalDeepDiveValidation

**Reasoning Complexity**: HIGH - Requires specialized domain knowledge and alternative proof approaches

**Recommended Models**: Claude Opus, GPT-4, or diverse strong model (different from Perspective 1)

**Note**: Using different models for Perspective 1 and 2 provides diverse validation perspectives.

---

### Tier 3: Synthesis (Cross-Cutting Analysis)

**Model**: `synthesis_model`

**Components:**
- **StrategySynthesizer**: Compare and merge dual proof strategies
- **AgentOverallAssessment** (both reviewers): Synthesize 4 parallel review components
- **Consensus Analysis**: Compare Gemini vs Codex reviews
- **ActionItems Generation**: Generate prioritized TODO list
- **Synthesis & Action Plan**: Final decision and roadmap
- **DetailedProofAgent**: Generate structured proof steps

**Reasoning Complexity**: VERY HIGH - Requires meta-reasoning, synthesis, and decision-making

**Recommended Models**: GPT-4, Claude Opus (slightly lower temperature for precise synthesis)

---

### Tier 4: Fast (Simple Extraction/Formatting)

**Model**: `fast_model`

**Components:**
- **ProofStatementAgent**: Formal/informal statement generation
- **CrossReferencesAgent**: Label extraction
- **FutureWorkAgent**: Gap categorization
- **AlternativeApproachesAgent**: Record rejected ideas
- **ExpansionRoadmapAgent**: Project planning
- **ValidationChecklistAgent**: Boolean checklist
- **DependencyLedgerAgent**: Dependency extraction
- **TechnicalDeepDiveAgent**: Challenge identification
- **Metadata Generator**: Report metadata
- **IterationFeedbackFormatter**: Feedback extraction

**Reasoning Complexity**: LOW - Simple extraction, formatting, and categorization

**Recommended Models**: GPT-3.5 Turbo, Claude Haiku, or other fast/cheap models

---

## Quick Start

### 1. Python API (Simplest)

```python
from mathster.proof_sketcher.llm_config import ProofSketcherLMConfig
from mathster.proof_sketcher.agent_factory import create_refine_pipeline

# Use a preset
config = ProofSketcherLMConfig.cost_optimized()

# Create pipeline
pipeline = create_refine_pipeline(lm_config=config, N=5, threshold=60)

# Run
result = pipeline(
    title_hint="KL Convergence Under LSI",
    theorem_label="thm-kl-convergence-lsi",
    theorem_type="MainResult",
    theorem_statement="...",
    document_source="docs/source/1_euclidean_gas/09_kl_convergence.md",
    creation_date="2025-01-12",
    proof_status="sketched",
)

print(f"Best score: {result.best_score:.2f}/100")
print(f"Iterations: {result.total_iterations}")
```

### 2. Hydra CLI (Most Flexible)

```bash
# Use default configuration
python examples/hydra_proof_sketcher.py

# Use cost-optimized preset
python examples/hydra_proof_sketcher.py llm=cost_optimized

# Override specific parameters
python examples/hydra_proof_sketcher.py \
    llm.perspective_1_model.model=gpt-4-turbo \
    llm.perspective_2_model.model=claude-3-opus-20240229 \
    pipeline.N=3

# Combine presets
python examples/hydra_proof_sketcher.py llm=development pipeline=quick
```

---

## Configuration Methods

### Method 1: Preset Configurations

```python
from mathster.proof_sketcher.llm_config import ProofSketcherLMConfig

# Balanced quality and cost
config = ProofSketcherLMConfig.default()

# Minimize cost with tiered models
config = ProofSketcherLMConfig.cost_optimized()

# Maximize quality (expensive)
config = ProofSketcherLMConfig.quality_optimized()

# Fast iteration for testing
config = ProofSketcherLMConfig.development()
```

### Method 2: Custom Configuration (Python)

```python
from mathster.proof_sketcher.llm_config import LLMModelConfig, ProofSketcherLMConfig

config = ProofSketcherLMConfig(
    perspective_1_model=LLMModelConfig(
        provider="openai",
        model="gpt-4-turbo-preview",
        temperature=0.7,
        max_tokens=4000,
    ),
    perspective_2_model=LLMModelConfig(
        provider="anthropic",
        model="claude-3-opus-20240229",
        temperature=0.7,
        max_tokens=4000,
    ),
    synthesis_model=LLMModelConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.5,  # Lower temperature for precise synthesis
        max_tokens=3000,
    ),
    fast_model=LLMModelConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=2000,
    ),
)

# Validate before use
config.validate()
```

### Method 3: YAML Configuration (Recommended for Hydra)

See [YAML Configuration](#yaml-configuration) section below.

---

## YAML Configuration

### Configuration File Structure

```
configs/
├── config.yaml                # Main Hydra config
├── llm/
│   ├── default.yaml          # Balanced configuration
│   ├── cost_optimized.yaml   # Cost-efficient tiering
│   ├── quality_optimized.yaml # Maximum quality
│   └── development.yaml      # Fast testing
└── pipeline/
    ├── default.yaml          # Standard refinement (N=5, threshold=60)
    └── quick.yaml            # Fast testing (N=2, threshold=50)
```

### Example: `configs/llm/cost_optimized.yaml`

```yaml
# Cost-optimized LLM configuration
# Uses strong models only for perspectives and synthesis

perspective_1_model:
  provider: "openai"
  model: "gpt-4-turbo-preview"
  temperature: 0.7
  max_tokens: 4000

perspective_2_model:
  provider: "anthropic"
  model: "claude-3-opus-20240229"
  temperature: 0.7
  max_tokens: 4000

synthesis_model:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.5  # Lower temp for synthesis
  max_tokens: 3000

fast_model:
  provider: "openai"
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 2000

# Claude models for tool calls (ReAct tools in strategists)
claude_model_heavy: "sonnet"
claude_model_fast: "haiku"
```

### Example: `configs/pipeline/default.yaml`

```yaml
# Default pipeline configuration

# Refinement parameters
N: 5                     # Maximum iterations
threshold: 60.0          # Score threshold (0-100)
fail_count: 5            # Consecutive failures before stopping

# Logging
verbosity: "STANDARD"    # MINIMAL, STANDARD, DETAILED, VERBOSE, DEBUG
log_json_path: null      # Optional JSON export path

# Feedback loop
enable_iteration_feedback: true  # Learn from previous iterations
```

---

## Python API

### Factory Functions

```python
from mathster.proof_sketcher.agent_factory import (
    create_drafting_agent,
    create_validator,
    create_sketch_pipeline,
    create_refine_pipeline,
)
from mathster.proof_sketcher.llm_config import ProofSketcherLMConfig

config = ProofSketcherLMConfig.cost_optimized()

# Create individual components
drafting_agent = create_drafting_agent(config)
validator = create_validator(config)

# Create full pipeline (no refinement)
pipeline = create_sketch_pipeline(config)

# Create refinement pipeline (recommended)
refine_pipeline = create_refine_pipeline(
    lm_config=config,
    N=5,
    threshold=60,
    verbosity="DETAILED",
)
```

### Direct Instantiation

```python
from mathster.proof_sketcher.llm_config import ProofSketcherLMConfig
from mathster.proof_sketcher.manual_refine_pipeline import (
    ManualRefineSketchPipeline,
    LogVerbosity,
)

config = ProofSketcherLMConfig.quality_optimized()

pipeline = ManualRefineSketchPipeline(
    lm_config=config,
    N=5,
    threshold=60.0,
    fail_count=5,
    verbosity=LogVerbosity.DETAILED,
    enable_iteration_feedback=True,
)

result = pipeline(**theorem_data)
```

---

## Hydra Integration

### Setup

1. **Install dependencies** (already in `pyproject.toml`):
   ```bash
   uv sync
   ```

2. **Create configuration files** (already in `configs/`):
   ```
   configs/
   ├── config.yaml
   ├── llm/*.yaml
   └── pipeline/*.yaml
   ```

3. **Run with Hydra**:
   ```bash
   python examples/hydra_proof_sketcher.py
   ```

### Command-Line Overrides

```bash
# Use preset
python examples/hydra_proof_sketcher.py llm=cost_optimized

# Override nested parameters
python examples/hydra_proof_sketcher.py \
    llm.perspective_1_model.temperature=0.8 \
    llm.synthesis_model.max_tokens=4000

# Override pipeline settings
python examples/hydra_proof_sketcher.py pipeline.N=10 pipeline.threshold=70

# Multiple overrides
python examples/hydra_proof_sketcher.py \
    llm=quality_optimized \
    pipeline=quick \
    pipeline.verbosity=DEBUG
```

### Custom Application

```python
import hydra
from omegaconf import DictConfig
from mathster.proof_sketcher.llm_config import ProofSketcherLMConfig
from mathster.proof_sketcher.agent_factory import create_refine_pipeline

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Load LLM config from Hydra
    lm_config = ProofSketcherLMConfig.from_dict(cfg.llm)

    # Validate
    lm_config.validate()

    # Create pipeline
    pipeline = create_refine_pipeline(
        lm_config=lm_config,
        N=cfg.pipeline.N,
        threshold=cfg.pipeline.threshold,
    )

    # Run your workflow
    result = pipeline(**your_data)

if __name__ == "__main__":
    main()
```

---

## Model Tier Recommendations

### For Production (Balanced Cost/Quality)

**Configuration**: `cost_optimized`

```yaml
perspective_1_model: gpt-4-turbo-preview    # Strong reasoning
perspective_2_model: claude-3-opus-20240229 # Diverse perspective
synthesis_model: gpt-4                       # Precise synthesis
fast_model: gpt-3.5-turbo                    # Cheap for simple tasks
```

**Estimated cost**: 40-50% reduction vs. using GPT-4 everywhere

**Use when**: Production proof generation with budget constraints

---

### For Critical Theorems (Maximum Quality)

**Configuration**: `quality_optimized`

```yaml
perspective_1_model: gpt-4-turbo-preview    # Maximum reasoning
perspective_2_model: claude-3-opus-20240229 # Diverse perspective
synthesis_model: gpt-4                       # Precise synthesis
fast_model: gpt-4                            # Quality everywhere
```

**Estimated cost**: 2-3x more than cost_optimized

**Use when**: Publication-ready proofs, critical theorems, final validation

---

### For Development/Testing

**Configuration**: `development`

```yaml
perspective_1_model: gpt-3.5-turbo  # Fast iteration
perspective_2_model: gpt-3.5-turbo  # Fast iteration
synthesis_model: gpt-3.5-turbo      # Fast iteration
fast_model: gpt-3.5-turbo           # Fast iteration
```

**Estimated cost**: ~10% of default configuration

**Use when**: Testing, debugging, CI/CD, rapid iteration

**Warning**: Lower quality - not suitable for production proofs

---

### For Mixed Workloads

**Configuration**: Custom

```yaml
# Heavy reasoning for perspectives
perspective_1_model: gpt-4-turbo-preview
perspective_2_model: gpt-4-turbo-preview

# Medium quality synthesis (acceptable tradeoff)
synthesis_model: gpt-4

# Cheap for simple tasks (largest volume)
fast_model: gpt-3.5-turbo
```

**Estimated cost**: 30% reduction vs. default

**Use when**: High proof volume with quality requirements

---

## Cost & Quality Tradeoffs

### Cost Breakdown by Tier (Approximate)

Assuming typical proof sketch workload:

| Tier | Model | Calls per Proof | Token Usage | Cost per Proof* |
|------|-------|-----------------|-------------|-----------------|
| **Perspective 1** | GPT-4 Turbo | ~8-12 | ~40k tokens | $1.20 |
| **Perspective 2** | Claude Opus | ~8-12 | ~40k tokens | $1.80 |
| **Synthesis** | GPT-4 | ~6-8 | ~20k tokens | $0.60 |
| **Fast** | GPT-3.5 | ~15-20 | ~25k tokens | $0.08 |
| **Total** | - | ~40-50 | ~125k tokens | **$3.68** |

*Approximate costs as of January 2025

### Configuration Comparison

| Configuration | Cost per Proof | Quality Score** | Use Case |
|---------------|----------------|-----------------|----------|
| **quality_optimized** | $5.50 | 92/100 | Critical theorems |
| **default** | $3.70 | 88/100 | Production |
| **cost_optimized** | $2.20 | 85/100 | High volume |
| **development** | $0.40 | 70/100 | Testing only |

**Quality scores are approximate based on validation metrics

### Optimization Strategies

1. **Start with default**: Good balance for most use cases
2. **Profile your workload**: Monitor which agents consume most tokens
3. **Optimize selectively**: Upgrade only the tiers that matter for your proofs
4. **Cache aggressively**: Use dspy's caching to avoid redundant calls
5. **Batch processing**: Process multiple theorems in parallel to amortize costs

---

## Advanced Usage

### Custom Model Providers

```python
from mathster.proof_sketcher.llm_config import LLMModelConfig, ProofSketcherLMConfig

config = ProofSketcherLMConfig(
    perspective_1_model=LLMModelConfig(
        provider="together",  # Together AI
        model="meta-llama/Llama-3-70b-chat-hf",
        temperature=0.7,
        max_tokens=4000,
        api_base="https://api.together.xyz/v1",  # Optional override
    ),
    # ... other tiers
)
```

### Provider-Specific Parameters

```python
config = ProofSketcherLMConfig(
    perspective_1_model=LLMModelConfig(
        provider="openai",
        model="gpt-4-turbo-preview",
        temperature=0.7,
        max_tokens=4000,
        additional_kwargs={
            "top_p": 0.95,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
        },
    ),
    # ... other tiers
)
```

### Dynamic Configuration

```python
def get_config_for_complexity(complexity: str) -> ProofSketcherLMConfig:
    """Select configuration based on theorem complexity."""
    if complexity == "simple":
        return ProofSketcherLMConfig.development()
    elif complexity == "medium":
        return ProofSketcherLMConfig.cost_optimized()
    elif complexity == "complex":
        return ProofSketcherLMConfig.quality_optimized()
    else:
        return ProofSketcherLMConfig.default()

# Use in workflow
theorem_complexity = assess_theorem_complexity(theorem)
config = get_config_for_complexity(theorem_complexity)
pipeline = create_refine_pipeline(lm_config=config)
```

### Debugging Configuration

```python
from omegaconf import OmegaConf

# Convert config to YAML for inspection
config = ProofSketcherLMConfig.cost_optimized()
print(OmegaConf.to_yaml(config))

# Validate configuration
try:
    config.validate()
    print("✓ Configuration valid")
except ValueError as e:
    print(f"✗ Configuration error: {e}")

# Inspect converted LMs
lms = config.to_dspy_lms()
for tier, lm in lms.items():
    print(f"{tier}: {lm}")
```

---

## Troubleshooting

### Common Issues

#### 1. "requires lm_config parameter"

**Error**: `TypeError: __init__() missing 1 required keyword-only argument: 'lm_config'`

**Cause**: All pipeline constructors now require explicit LLM configuration.

**Solution**:
```python
# ✗ Wrong (old code)
pipeline = AgentSketchPipeline()

# ✓ Correct
from mathster.proof_sketcher.llm_config import ProofSketcherLMConfig
config = ProofSketcherLMConfig.default()
pipeline = AgentSketchPipeline(lm_config=config)
```

#### 2. Invalid Model Configuration

**Error**: `ValueError: perspective_1_model.temperature must be in [0.0, 2.0]`

**Cause**: Configuration validation failed.

**Solution**: Check temperature, max_tokens, and other parameters:
```python
config = ProofSketcherLMConfig.default()
config.validate()  # Will raise ValueError with specific issue
```

#### 3. API Key Not Found

**Error**: `openai.AuthenticationError: No API key provided`

**Cause**: Missing environment variables.

**Solution**: Set API keys in environment:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or use `.env` file:
```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

#### 4. Hydra Configuration Not Found

**Error**: `ConfigNotFoundError: Cannot find primary config 'config'`

**Cause**: Running from wrong directory or missing config files.

**Solution**:
```bash
# Run from project root
cd /path/to/fragile
python examples/hydra_proof_sketcher.py

# Or specify config path explicitly
python examples/hydra_proof_sketcher.py --config-path=../configs
```

#### 5. Model Not Available

**Error**: `BadRequestError: Model 'gpt-4-turbo-preview' does not exist`

**Cause**: Model name incorrect or not accessible with your API key.

**Solution**: Check model availability and update configuration:
```yaml
perspective_1_model:
  provider: "openai"
  model: "gpt-4-1106-preview"  # Use correct model name
```

---

## Best Practices

1. **Always validate configuration** before creating pipelines:
   ```python
   config.validate()
   ```

2. **Use presets for consistency** across your team:
   ```python
   config = ProofSketcherLMConfig.cost_optimized()
   ```

3. **Version control your YAML configs** for reproducibility:
   ```bash
   git add configs/llm/production.yaml
   ```

4. **Monitor costs** with logging:
   ```python
   pipeline = create_refine_pipeline(
       lm_config=config,
       verbosity="DETAILED",  # Logs model usage
   )
   ```

5. **Test with development preset** before production:
   ```python
   # Test logic with cheap models
   dev_config = ProofSketcherLMConfig.development()
   test_pipeline = create_refine_pipeline(lm_config=dev_config)

   # Production run
   prod_config = ProofSketcherLMConfig.cost_optimized()
   pipeline = create_refine_pipeline(lm_config=prod_config)
   ```

6. **Use Hydra for experiment tracking**:
   ```bash
   # Each run saves config and outputs to timestamped directory
   python examples/hydra_proof_sketcher.py llm=cost_optimized
   # Outputs saved to: outputs/2025-01-12/14-30-00/
   ```

---

## Further Reading

- **dspy Documentation**: https://dspy-docs.vercel.app/
- **Hydra Documentation**: https://hydra.cc/docs/intro/
- **OmegaConf Documentation**: https://omegaconf.readthedocs.io/
- **Module Source Code**: `src/mathster/proof_sketcher/`
  - `llm_config.py` - Configuration dataclasses
  - `agent_factory.py` - Factory functions
  - `sketch_pipeline.py` - Pipeline orchestration
  - `manual_refine_pipeline.py` - Refinement loop

---

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section above
2. Review example code in `examples/hydra_proof_sketcher.py`
3. Inspect YAML configs in `configs/llm/`
4. File an issue with reproduction steps

---

**Last Updated**: 2025-01-12
**Version**: 1.0.0
