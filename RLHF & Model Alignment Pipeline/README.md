# RLHF & Model Alignment Pipeline

## Overview
A complete implementation of Reinforcement Learning from Human Feedback (RLHF) using Direct Preference Optimization (DPO) to align Large Language Models with human preferences, featuring parameter-efficient fine-tuning and comprehensive evaluation.

## Architecture
- **Base Model**: Qwen2.5-1.5B-Instruct (quantized for efficiency)
- **Alignment Method**: Direct Preference Optimization (DPO) over traditional RLHF
- **Parameter Efficiency**: QLoRA (4-bit quantization + LoRA adapters)
- **Data Pipeline**: UltraFeedback preference dataset processing
- **Evaluation**: Automated testing with preference margin metrics

## Tech Stack
- **Python** - Core implementation
- **Transformers** - Model loading and tokenization
- **TRL (Transformer Reinforcement Learning)** - DPO training framework
- **PEFT (Parameter-Efficient Fine-Tuning)** - LoRA adapter implementation
- **BitsAndBytes** - 4-bit quantization for memory efficiency
- **Weights & Biases** - Training monitoring and logging
- **Datasets** - Hugging Face dataset management

## How to Run

1. **Install dependencies**:
   ```bash
   pip install datasets accelerate peft bitsandbytes wandb
   pip install git+https://github.com/huggingface/trl.git
   ```

2. **Set up Weights & Biases** (optional):
   ```bash
   wandb login
   ```

3. **Run DPO training**:
   ```bash
   python dpo.py
   ```

4. **Or use notebook**:
   ```bash
   jupyter notebook dpo.ipynb
   ```

## Key Decisions
- **DPO over PPO**: Stable preference learning without separate reward model
- **QLoRA Approach**: 4-bit quantization + LoRA for memory-efficient fine-tuning
- **Small Model**: Qwen2.5-1.5B chosen for demonstration and resource constraints
- **Preference Dataset**: UltraFeedback for high-quality human preference data
- **Short Training**: Limited steps for demo while showing convergence

## Training Pipeline
1. **Model Quantization**: 4-bit loading to reduce VRAM requirements
2. **LoRA Configuration**: Attention layers adaptation for task-specific learning
3. **Preference Data**: Convert RLHF triples to DPO format (prompt, chosen, rejected)
4. **DPO Training**: Implicit reward optimization through preference pairs
5. **Adapter Saving**: Modular fine-tuned components for deployment

## Technical Advantages
- **Memory Efficient**: QLoRA enables training on consumer GPUs
- **Stable Training**: DPO avoids complex RL optimization issues
- **Modular Design**: LoRA adapters can be swapped for different alignments
- **Implicit Rewards**: No separate reward model training required
- **Fast Convergence**: Direct optimization of preference objectives

## Evaluation Metrics
- **Preference Margins**: Difference in log probabilities between chosen/rejected responses
- **Training Loss**: DPO objective convergence monitoring
- **Response Quality**: Human preference alignment assessment
- **Model Stability**: Comparison with base model behavior

## Use Cases
- **Safety Alignment**: Reducing harmful or biased responses
- **Helpfulness Tuning**: Improving response quality and relevance
- **Style Adaptation**: Matching specific communication preferences
- **Domain Specialization**: Adapting to particular use case requirements
