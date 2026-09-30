# nanoGPT 101

A visual, matrix-by-matrix explanation of a tiny GPT forward pass.

This project is inspired by the readable architecture of [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT), but the browser model here is intentionally tiny and deterministic so every number can be inspected.

## What is computed

For the fixed sequence `The cat sat on the mat`, the app computes:

```text
X = token_embeddings + position_embeddings
Q = X Wq, K = X Wk, V = X Wv
A = softmax(mask(QK^T / sqrt(d)))
C = A V
H = (X + C) + W2 ReLU(W1 (X + C))
logits = H Wout
P(next token) = softmax(logits / temperature)
```

The matrix values live in `src/lib/tinyGpt.ts`. The page in `src/pages/index.tsx` renders the intermediate matrices and lets you inspect rows, weights, logits, and probabilities.

## Run it

```powershell
npm install
npm run dev
```

Open `http://localhost:3000`.

This is an educational forward-pass visualizer, not a production language model trainer. The next extension should add a small training loop and show how changing the weights changes the prediction.

## Real GPT-2 checkpoint

The **Load real GPT-2** control downloads the `Xenova/gpt2` ONNX conversion in the browser through the Xenova Transformers runtime. It shows the real GPT-2 BPE token IDs and the real next-token logits for the example sentence. The compact matrix flow remains available alongside it as an inspectable teaching model.
