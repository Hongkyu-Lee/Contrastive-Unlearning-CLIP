## Contrastive Unlearning for Few-shot classifiers

This is a code repository for ICLR2025 rebuttal.
This repository contains code for unlearning specific classes from CLIP models while retaining performance on other classes. The key components are:

### Main Files
- `unlearn.py`: Contains the core unlearning implementation including:
  - Contrastive unlearning loss function
  - Fine-tuning and retention mechanisms
  - Zero-shot evaluation
  - Training loops for unlearning and retention

- `finetune.py`: Contains the fine-tuning implementation including:
  - Dataset wrapper for image-text pairs
  - Fine-tuning training loop with contrastive loss
  - Zero-shot evaluation on test set
  - Model checkpointing to save best performing model
  - Wandb integration for experiment tracking


### Requirements
- PyTorch
- Weights & Biases
- Torchvision
- TQDM
- PIL
