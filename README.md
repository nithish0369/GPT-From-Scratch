# GPT from Scratch: A Minimal PyTorch Implementation  
**Build a Generative Pre-trained Transformer (GPT) model from the ground up**, following Andrej Karpathy's tutorial [Let's build GPT](https://youtu.be/kCc8FmEb1nY). This project implements a decoder-only transformer architecture for **character-level text generation** using PyTorch.  

## 🔑 Features  
- **Transformer Architecture**:  
  - Multi-head self-attention with causal masking  
  - Position-wise feed-forward networks  
  - Residual connections & layer normalization  
- **Tokenization**: Byte-level encoding/decoding  
- **Training**:  
  - AdamW optimizer with cosine learning rate decay  
  - Gradient checkpointing for memory efficiency  
- **Inference**:  
  - Temperature-controlled sampling  
  - Top-k filtering for diverse text generation  
- **Scalability**: Supports nanoGPT-style models (~10M params).

- ## ⚙️ Installation  
**Dependencies**:  
```bash
pip install torch numpy tqdm

python data/shakespeare/prepare.py  # Downloads and preprocesses Shakespeare data


---

#### **4. Usage**  
**Training**:  
```markdown
## 🚀 Usage  
### Train the model:  
```bash  
python train.py \  
  --dataset=shakespeare \  
  --batch_size=64 \  
  --block_size=256 \  
  --n_embd=384 \  
  --n_head=6 \  
  --n_layer=6 \  
  --max_iters=5000


**Text Generation**:  
```markdown
### Generate text:  
```bash  
python generate.py \  
  --prompt="KING RICHARD III:" \  
  --temperature=0.9 \  
  --top_k=40 \  
  --max_new_tokens=500


---

#### **5. Code Structure**  
```markdown
## 📂 Code Structure

## 📊 Results  
**Shakespeare Generation** (after 5K steps):  
```text  
KING RICHARD III:  
Now is the winter of our discontent  
Made glorious summer by this sun of York;  
And all the clouds that lour'd upon our house  
In the deep bosom of the ocean buried.


---

#### **7. References & Credits**  
```markdown
## 📚 References  
1. [Attention Is All You Need (Vaswani et al.)](https://arxiv.org/abs/1706.03762)  
2. Andrej Karpathy: [nanoGPT](https://github.com/karpathy/nanoGPT) & [Video Tutorial](https://youtu.be/kCc8FmEb1nY)  
3. Dataset: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

## 📜 License  
MIT License. See [LICENSE](LICENSE) for details.

