# City Name Language Models

This repository contains small notebook experiments for generating city-like names
from `worldcities.csv`.

The notebooks progress through three character-level models:

- `bigram.ipynb`: predicts the next character from the previous character.
- `trigram.ipynb`: predicts the next character from the previous two characters.
- `mlp.ipynb`: trains a small PyTorch neural network over fixed-length character context.

These changes were added as simple diff material and can be reverted later.
