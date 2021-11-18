# qtransformer

qTransformer is a quantum circuit neural network classifier based on chained quantum attention mechanism layers, residual layers, and feed-forward neural network layers. Attention is computed using a trainable, parameterized general two-body interaction between word embedding statevectors in a sentence system; residual connections are represented using CX-gates; feed-forward neural network layers are represented using custom variational ansatzes such as RealAmplitudes and EfficientSU2.

The construction for the circuit representing the model is as follows:
 ```python
 # ------------------------------------------------------------------------------
 # Construct Circuit.

 # Feature map encodes the word embedding vectors.
 feature_map = PauliFeatureMap(n)

 # Attention transforms the sentence system to contain information regarding how
 # much each word qubit should attend to every other word qubit.
 if include_attention:
     attention_layers = [attention(n, l=i) for i in range(reps)]

 # Residual adds information regarding the original word emebddings back to the
 # post-attention sentence system.
 if include_residual:
     residual_layers = [residual(n) for _ in range(reps)]

 # Ansatz serves as a feed-forward layer at the end of the transformer block.
 feed_forward_layers = [
     EfficientSU2(n, parameter_prefix=str(i)) for i in range(reps)
 ]

 self.circuit = QuantumCircuit(2 * n if include_residual else n)

 self.circuit.append(feature_map, range(n))
 if include_residual:
     self.circuit.append(feature_map, range(n, 2 * n))

 for i in range(reps):
     if include_attention:
         self.circuit.append(attention_layers[i], range(n))
     if include_residual:
         self.circuit.append(residual_layers[i], range(2 * n))
     self.circuit.append(feed_forward_layers[i], range(n))
 # ------------------------------------------------------------------------------
```
 

The wrapper class for the qTransformer lives in `/qtransformer/__init__.py`.
