# qtransformer

qTransformer is a quantum circuit neural network classifier based on chained quantum attention mechanism layers, residual layers, and feed-forward neural network layers. Attention is computed using a trainable, parameterized general two-body interaction between word embedding statevectors in a sentence system; residual connections are represented using CX-gates; feed-forward neural network layers are represented using custom variational ansatzes such as RealAmplitudes and EfficientSU2.

The wrapper class for the qTransformer lives in `/qtransformer/__init__.py`.
