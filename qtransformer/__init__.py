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
    """QuantumTransformer Wrapper Class

    An object representing a quantum transformer. The constructor builds a quantum
    circuit with the specified parameters, a CircuitQNN from the circuit, and a
    NeuralNetworkClassifier from the QNN. Data can be pre-processed using the model
    parameters, and the model can be trained to a given corpus using the train
    method. The score method returns the accuracy of the model on a labelled corpus,
    and the predict method returns the predicted sentiment classification for a given
    sentence.
    """

    def embed_sentence(self, sentence):
        """Returns the embedding vector for a given sentence (rep. as a list of
           words). Sentence embeddings are of fixed length sequence_length and
           are padded with zeros if they aren't long enough.

        Args:
            sentence (string): The string to embed.

        Returns:
            [np.array(dtype=float64)]: The embedding vector representation of the
                                       string of dimension specified as a constructor
                                       parameter.
        """
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
        """Builds a quantum circuit with the specified parameters, a CircuitQNN from the
           circuit, and a NeuralNetworkClassifier from the QNN.

        Args:
            sequence_length (int):              The number of words in each sentence.
            embedding_dimension (int):          The dimension of the word embeddings.
            embedding_dict (dict):              A dictionary mapping from word to
                                                embedding vector.

            quantum_instance (QuantumInstance): The quantum instance on which to
                                                execute the NeuralNetworkClassifier
                                                circuit.
            reps (int):                         The number of transformer layers in the
                                                model.
            iterations (int):                   The max number of iterations for which
                                                the NeuralNetworkClassifier is trained.
            include_attention (boolean):        Whether to include attention layers in
                                                the model.
            include_residual (boolean):         Whether to include residual layers in
                                                the model.
        """

        def attention(n, l=0):
            """Computes the attention between every pair of words embedding qubits in
               the n-qubit sentence by performing the attention pair subroutine on every
               pair.

            Args:
                n (int): The number of word embedding qubits in the circuit repesenting
                         the sentence.
                l (int): The index of the current attention layer.

            Returns:
                [Gate]: A gate that transforms the sentence circuit to contain
                        information regarding the attention for every pair of word
                        embeddings.
            """

            def attention_pair(i, j, l=0):
                """Transforms the state to contain information regarding the attention
                   for a pair of word embeddings (of dimension N) represented as
                   statevectors for an N-qubit system using a trainable parameterized
                   general two-body interaction.

                Args:
                    i, j (int): The indices of the pair of word embedding qubits.

                Returns:
                    [Gate]: A gate that takes two qubits and transforms the two-qubit
                            state to contain information regarding how much each word
                            should attend to each other.
                """

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

            qc = QuantumCircuit(n)
            for pair in itertools.combinations(range(n), 2):
                qc.append(attention_pair(*pair, l=l), pair)
            return qc.to_gate(label="Attention")

        def residual(n):
            """Adds the residual of the word embeddings to the post-attention sentence
               circuit state via CX-gates (i.e. addition % 2).

            Args:
                n (int): The number of word embedding qubits in the circuit repesenting
                         the sentence.

            Returns:
                [Gate]: A gate that adds the residual of the original word embeddings
                        to the post-attention sentence circuit state.
            """

            qc = QuantumCircuit(2 * n)
            for i in range(n):
                qc.cx(n + i, i)
            return qc.to_gate(label="Residual")

        # Classifier Utilities
        def parity(x):
            """The interpreter function for the NeuralNetworkClassifier that returns
            the parity of the output.
            """
            return "{:b}".format(x).count("1") % 2

        def callback_graph(weights, obj_func_eval):
            """The live callback graph displaying the squared error over time (in
            iterations).
            """
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

        # ------------------------------------------------------------------------------
        # Reduces the dimensionality of the emebddings to the desired embedding
        # dimension.

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
        # ------------------------------------------------------------------------------

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

        # ------------------------------------------------------------------------------
        # Circuit --> Network.
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
        # ------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------
        # Network --> Classifier.
        self.classifier = NeuralNetworkClassifier(
            neural_network=self.network,
            optimizer=COBYLA(maxiter=iterations),
            callback=callback_graph,
        )
        # ------------------------------------------------------------------------------

    def preprocess_data(self, corpus):
        """Pre-process the corpus by filtering out sentences with words not contained in
           the embeddings dictionary and taking sub-sequences of length sequence_length
           that are of length greater than 2. Convert words to embedding vectors. Return
           the X and Y matrices that the model can be trained on.

        Args:
            corpus ([{"sentence": ..., "sentiment": ...}, ...]): A list of dictionaries
                                                                 each with a sentence
                                                                 and a sentiment.

        Returns:
            X (np.array): An array of (arrays of (embedding vectors representing words)
                          representing sentences) representing the processed corpus.
            Y (np.array): An array of sentiment classifications (1 for positive and 0
                          for negative).
        """

        # ------------------------------------------------------------------------------
        # Filter out sentences with words not contained in the embeddings dictionary
        # and taking sub-sequences of length sequence_length that are of length greater
        # than 2.
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
        # ------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------
        # Determine the vocabulary for the corpus.
        vocab = []
        for sequence in filtered_sequences:
            vocab.extend(sequence["sentence"])
        vocab = list(set(vocab))
        # ------------------------------------------------------------------------------

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
        """Train the classifier on the given labelled corpus.

        Args:
            X (np.array): An array of (arrays of (embedding vectors representing words)
                          representing sentences) representing the processed corpus.
            Y (np.array): An array of sentiment classifications (1 for positive and 0
                          for negative).
        """
        plt.rcParams["figure.figsize"] = (12, 6)
        self.classifier.fit(X, Y)
        plt.rcParams["figure.figsize"] = (6, 4)

    def score(self, X, Y):
        """Compute the accuracy of the predicted sentiment classifications for
           the given labelled corpus.

        Args:
            X (np.array): An array of (arrays of (embedding vectors representing words)
                          representing sentences) representing the processed corpus.
            Y (np.array): An array of sentiment classifications (1 for positive and 0
                          for negative).

        Returns:
            score (int): The accuracy of the predicted classifications.
        """
        return self.classifier.score(X, Y)

    def predict(self, sentence):
        """Predict the sentiment classification for the given sentence.

        Args:
            sentence (string): The sentence to classify.

        Returns:
            sentiment (np.array): The sentiment classification for the sentence.
        """
        return self.classifier.predict(
            self.embed_sentence(
                [
                    "".join(e for e in word if e.isalnum()).lower()
                    for word in sentence.split()
                ]
            )
        )
