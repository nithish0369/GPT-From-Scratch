# GPT-From-Scratch

"A minimal PyTorch implementation of a GPT-like transformer model, built from scratch based on Andrej Karpathy's tutorial. Generates text character-by-character."

Key Goal: Demonstrate understanding of transformer architecture, self-attention, and autoregressive language modeling.

Features

✅ Core Components:

Byte-level tokenization

Transformer blocks with multi-head self-attention

Positional embeddings

Autoregressive text generation

✅ Training:

Trained on Shakespeare/TinyStories datasets

AdamW optimizer with cosine learning rate decay

✅ Inference:

Text generation via temperature-controlled sampling

Code Highlights

Link to critical files:

model.py: Transformer architecture (attention heads, feed-forward layers)

train.py: Training loop with gradient checkpointing

generate.py: Text sampling script

Add code snippets for key functions (e.g., self_attention, generate()).

Results

Include generated text samples (e.g., Shakespearean prose or original text).

Example:

text
"KING RICHARD III:  
Now is the winter of our discontent..."  
Add training metrics (e.g., val_loss: 0.123).

Setup

Dependencies: Python 3.10+, PyTorch 2.0+

How to train: python train.py --dataset=shakespeare

How to generate: python generate.py --prompt="Once upon a time"

References

"Inspired by Andrej Karpathy's Let’s build GPT"

Paper: Attention Is All You Need (Vaswani et al.)
