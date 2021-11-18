import numpy as np
import matplotlib.pyplot as plt

from qiskit import Aer, QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.parametertable import ParameterView
from qiskit.circuit.library import PauliFeatureMap, EfficientSU2
from qiskit.algorithms.optimizers import COBYLA

from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

from sklearn.decomposition import PCA

import itertools

from IPython.display import clear_output


class QuantumTransformer:
    def embed_sentence(self, sentence):
        embedded_sentence = np.array(
            [self.embeddings[word] for word in sentence]
        ).flatten()
        return np.pad(
            embedded_sentence, (0, self.sequence_length - embedded_sentence.size)
        )

    def __init__(
        self,
        sequence_length,
        embedding_dimension,
        embedding_dict,
        quantum_instance=Aer.get_backend("qasm_simulator"),
        reps=1,
        iterations=100,
        include_attention=True,
        include_residual=True,
    ):

        # Model Layers
        def attention_pair(i, j, l=0):
            qc = QuantumCircuit(2)
            qc.u(*ParameterVector("atn_%d_pair_%d%d_0" % (l, i, j), length=3), 0)
            qc.u(*ParameterVector("atn_%d_pair_%d%d_1" % (l, i, j), length=3), 1)
            qc.cx(1, 0)
            qc.u(*ParameterVector("atn_%d_pair_%d%d_2" % (l, i, j), length=3), 0)
            qc.u(*ParameterVector("atn_%d_pair_%d%d_3" % (l, i, j), length=3), 1)
            qc.cx(0, 1)
            qc.u(*ParameterVector("atn_%d_pair_%d%d_4" % (l, i, j), length=3), 0)
            qc.u(*ParameterVector("atn_%d_pair_%d%d_5" % (l, i, j), length=3), 1)
            qc.cx(1, 0)
            qc.u(*ParameterVector("atn_%d_pair_%d%d_6" % (l, i, j), length=3), 0)
            qc.u(*ParameterVector("atn_%d_pair_%d%d_7" % (l, i, j), length=3), 1)
            return qc.to_gate(label="Attention(%d, %d)" % (i, j))

        def attention(n, l=0):
            qc = QuantumCircuit(n)
            for pair in itertools.combinations(range(n), 2):
                qc.append(attention_pair(*pair, l=l), pair)
            return qc.to_gate(label="Attention")

        def residual(n):
            qc = QuantumCircuit(2 * n)
            for i in range(n):
                qc.cx(n + i, i)
            return qc.to_gate(label="Residual")

        # Classifier Utilities
        def parity(x):
            return "{:b}".format(x).count("1") % 2

        def callback_graph(weights, obj_func_eval):
            clear_output(wait=True)
            objective_func_vals.append(obj_func_eval)
            plt.title("Objective Function vs. Iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Objective Function")
            plt.plot(range(len(objective_func_vals)), objective_func_vals)
            plt.show()

        self.sequence_length = sequence_length
        self.embedding_dimension = embedding_dimension

        n = self.sequence_length * self.embedding_dimension

        pca = PCA(n_components=self.embedding_dimension)
        self.embeddings = {}
        word_to_num = {}
        embeds_matrix = []

        for i, word in enumerate(embedding_dict.vocab.keys()):
            word_to_num[word] = i
            embeds_matrix.append(embedding_dict[word])

        embeds_matrix = np.array(embeds_matrix)
        embeds_matrix = pca.fit_transform(embeds_matrix)
        for word in embedding_dict.vocab.keys():
            self.embeddings[word] = embeds_matrix[word_to_num[word]]

        feature_map = PauliFeatureMap(n)
        if include_attention:
            attention_layers = [attention(n, l=i) for i in range(reps)]
        if include_residual:
            residual_layers = [residual(n) for _ in range(reps)]
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

        self.network = CircuitQNN(
            circuit=self.circuit,
            input_params=feature_map.parameters,
            weight_params=ParameterView(
                (
                    [parameter for l in attention_layers for parameter in l.params]
                    if include_attention
                    else []
                )
                + [parameter for l in feed_forward_layers for parameter in l.parameters]
            ),
            interpret=parity,
            output_shape=2,
            quantum_instance=quantum_instance,
        )

        self.classifier = NeuralNetworkClassifier(
            neural_network=self.network,
            optimizer=COBYLA(maxiter=iterations),
            callback=callback_graph,
        )

    def preprocess_data(self, corpus):
        filtered_sequences = []
        for sentence in corpus:
            sequences = [
                {
                    "sentence": sentence["sentence"][i : i + self.sequence_length],
                    "sentiment": sentence["sentiment"],
                }
                for i in range(0, len(sentence["sentence"]), self.sequence_length)
            ]
            for sequence in sequences:
                if all(word in self.embeddings for word in sequence["sentence"]):
                    filtered_sequences.append(sequence)
        filtered_sequences = [
            sentence for sentence in filtered_sequences if len(sentence["sentence"]) > 2
        ]

        vocab = []
        for sequence in filtered_sequences:
            vocab.extend(sequence["sentence"])
        vocab = list(set(vocab))

        X = np.array(
            [
                self.embed_sentence(sentence["sentence"])
                for sentence in filtered_sequences
            ]
        )
        Y = np.array(
            [
                sentence["sentiment"]
                for sentence in filtered_sequences
                if len(sentence["sentence"]) > 2
            ]
        )

        return X, Y

    def train(self, X, Y):
        plt.rcParams["figure.figsize"] = (12, 6)
        self.classifier.fit(X, Y)
        plt.rcParams["figure.figsize"] = (6, 4)

    def score(self, X, Y):
        return self.classifier.score(X, Y)

    def predict(self, sentence):
        print()
        return self.classifier.predict(
            self.embed_sentence(
                [
                    "".join(e for e in word if e.isalnum()).lower()
                    for word in sentence.split()
                ]
            )
        )
