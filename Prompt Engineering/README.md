# Sarcasm Detection with Prompt Engineering

## Overview
A comprehensive study exploring how different prompt engineering techniques can improve Large Language Model performance on the challenging task of sarcasm detection, using a Reddit sarcasm dataset and fine-tuned LLaMA model.

## Architecture
- **Dataset**: Reddit comments labeled for sarcasm (balanced training set)
- **Model**: Fine-tuned LLaMA 3.1 8B model specialized for sarcasm detection
- **Prompt Techniques**: Multiple approaches tested (Few-Shot, Role, CoT, Step-Back, etc.)
- **Evaluation**: Accuracy, precision, recall, F1-score across prompting methods
- **Context Integration**: Parent comment context for conversational sarcasm detection

## Tech Stack
- **Python** - Core implementation
- **Llama.cpp** - Efficient LLM inference (GGUF format)
- **Hugging Face** - Model downloading and tokenization
- **pandas** - Data manipulation and results analysis
- **scikit-learn** - Evaluation metrics
- **Kaggle API** - Dataset access

## Prompting Techniques Tested
- **Few-Shot**: Labeled examples to teach sarcasm patterns
- **Role Prompting**: Model acts as sarcasm detection specialist
- **Chain-of-Thought**: Step-by-step reasoning process
- **Step-Back**: General context description before classification
- **Rephrase-and-Respond**: Literal rephrasing to expose contradictions
- **Chain-of-Verification**: Self-questioning verification process
- **Self-Refine**: Model critiques and improves its own answers

## How to Run

1. **Install dependencies**:
   ```bash
   pip install kaggle scikit-learn transformers torch llama-cpp-python
   ```

2. **Set up Kaggle API**:
   ```bash
   # Download kaggle.json from your Kaggle account
   # Place in working directory
   chmod 600 kaggle.json
   ```

3. **Download dataset**:
   ```bash
   kaggle datasets download -d danofer/sarcasm
   unzip sarcasm.zip -d sarcasm_data
   ```

4. **Run analysis**:
   ```bash
   python prompts_evaluation.py
   ```

5. **Or use notebook**:
   ```bash
   jupyter notebook prompts_evaluation.ipynb
   ```

## Key Decisions
- **Fine-tuned Model**: Used sarcasm-specialized LLaMA instead of general-purpose model
- **Context Importance**: Included parent comments for conversational context
- **Text Preservation**: Kept punctuation, capitalization, and emojis as sarcasm cues
- **Multiple Techniques**: Systematic comparison of prompting approaches
- **Resource Constraints**: Focused on prompt engineering rather than full fine-tuning

## Results Analysis
- **Context Critical**: Parent comment context significantly improves detection accuracy
- **Chain-of-Verification**: Most effective technique for catching nuanced sarcasm
- **Common Pitfalls**: Over-simplification in rephrasing, loss of conversational context
- **Performance Gains**: Advanced prompting techniques show measurable improvements over basic classification

## Key Findings
1. Sarcasm detection requires both contextual understanding and tone analysis
2. Self-verification techniques (CoV) outperform single-pass classification
3. Preserving conversational context is crucial for accurate detection
4. Different prompting techniques excel at different types of sarcasm patterns
