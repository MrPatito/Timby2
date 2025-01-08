import os
import warnings
import numpy as np
import pandas as pd
import time
from datetime import datetime
import concurrent.futures
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Dense, Attention, LayerNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from neo4j import GraphDatabase
import tensorflow as tf

# Suppress TensorFlow and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class DeepSequenceEmbedder:
    """
    A flexible deep learning model for sequence embedding.
    """
    def __init__(self, vocab_size, embedding_dim, sequence_length,
                 rnn_units=128, rnn_type='LSTM', num_heads=4,
                 ff_dim=128, dropout_rate=0.1, dense_units=128):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.rnn_units = rnn_units
        self.rnn_type = rnn_type
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=(self.sequence_length,))
        embedding_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)(inputs)
        x = embedding_layer

        # Recurrent layers
        if self.rnn_type == 'LSTM':
            rnn_layer = LSTM(self.rnn_units, return_sequences=True)
        elif self.rnn_type == 'GRU':
            rnn_layer = GRU(self.rnn_units, return_sequences=True)
        else:
            raise ValueError("Unsupported RNN type. Choose 'LSTM' or 'GRU'.")
        x = rnn_layer(x)

        # Attention mechanism
        query = Dense(self.rnn_units)(x)
        key = Dense(self.rnn_units)(x)
        value = Dense(self.rnn_units)(x)
        attention = Attention()([query, key, value])
        x = LayerNormalization(epsilon=1e-6)(x + attention)

        # Feed-forward network
        ffn_output = Dense(self.ff_dim, activation="relu")(x)
        ffn_output = Dropout(self.dropout_rate)(ffn_output)
        ffn_output = Dense(self.rnn_units)(ffn_output)
        x = LayerNormalization(epsilon=1e-6)(x + ffn_output)

        # Pooling layer
        pooled_output = tf.reduce_mean(x, axis=1)

        # Dense layers for final embedding
        dense_1 = Dense(self.dense_units, activation='relu')(pooled_output)
        outputs = Dense(self.dense_units)(dense_1)

        return Model(inputs=inputs, outputs=outputs)

class SequenceDataProcessor:
    """
    Handles loading and processing sequence data.
    """
    def __init__(self, csv_path, sequence_length, overlap_factor=0.5):
        self.csv_path = csv_path
        self.sequence_length = sequence_length
        self.overlap_step = int(sequence_length * (1 - overlap_factor))
        self.data = self._load_data()

    def _load_data(self):
        """Loads data from CSV."""
        try:
            return pd.read_csv(self.csv_path)
        except Exception as e:
            raise FileNotFoundError(f"Error loading data from {self.csv_path}: {e}")

    def generate_sequences(self):
        """Generates sequences from the loaded data."""
        sequences = []
        results = []
        number_data = self.data['Num'].tolist()
        for i in range(0, len(number_data) - self.sequence_length, self.overlap_step):
            sequences.append(number_data[i:i + self.sequence_length])
            results.append(number_data[i + self.sequence_length])
        return np.array(sequences), np.array(results)

    def prepare_for_embedding(self, sequences):
        """Prepares sequences for embedding (e.g., padding, encoding)."""
        return np.array(sequences)

class SequenceSimilaritySearch:
    """
    Generates embeddings and performs similarity searches.
    """
    def __init__(self, embedder, graph_db):
        self.embedder = embedder
        self.graph_db = graph_db

    def generate_embeddings(self, sequences):
        """Generates embeddings for a set of sequences."""
        prepared_sequences = self.embedder.model.predict(sequences)
        return prepared_sequences

    def store_embeddings_neo4j(self, sequences, embeddings, results):
        """Stores sequences, embeddings, and results in Neo4j."""
        with self.graph_db.driver.session() as session:
            for seq, emb, res in zip(sequences, embeddings, results):
                self._store_sequence_transaction(session, seq.tolist(), res, emb.tolist())

    def _store_sequence_transaction(self, session, sequence, result, embedding):
        """Transaction to store sequence data in Neo4j."""
        query = """
        CREATE (s:NewSequence {
            timestamp: timestamp(),
            numbers: $sequence,
            embedding: $embedding
        })
        WITH s
        MATCH (n:Number {value: $result_number})
        CREATE (s)-[:NEXT_IS]->(n)
        """
        session.run(query, sequence=sequence, embedding=embedding, result_number=int(result))

    def find_similar_sequences(self, query_sequence, limit=5):
        """Finds similar sequences in Neo4j based on embeddings."""
        query_embedding = self.embedder.model.predict(np.array([query_sequence]))
        with self.graph_db.driver.session() as session:
            return self._find_similar_sequences_transaction(session, query_embedding.tolist()[0], limit)

    def _find_similar_sequences_transaction(self, session, query_embedding, limit):
        """Transaction to find similar sequences using cosine similarity."""
        query = """
        MATCH (s:NewSequence)-[:NEXT_IS]->(r:Number)
        WITH s, r,
             // Calculate cosine similarity in Python instead of using GDS
             // You may need to fetch embeddings and calculate similarity here
        """
        # Execute the query and process results accordingly
        # ...
        return []  # Ensure to return an empty list if no results are found

class RouletteGraphDB:
    """Manages connection and basic operations with Neo4j."""
    def __init__(self, uri, user, password):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = self._connect()

    def _connect(self):
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            driver.verify_connectivity()
            print("Successfully connected to Neo4j.")
            return driver
        except Exception as e:
            raise Exception(f"Could not connect to Neo4j: {e}")

    def close(self):
        if self.driver:
            self.driver.close()

# Example Usage
if __name__ == "__main__":
    # --- Configuration ---
    CSV_FILE = "secuencia_ruleta_V2.csv"
    SEQUENCE_LENGTH = 50
    EMBEDDING_DIM = 100
    VOCAB_SIZE = 37 # Numbers 0-36
    RNN_UNITS = 128
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "123456789"

    # --- Data Processing ---
    data_processor = SequenceDataProcessor(CSV_FILE, SEQUENCE_LENGTH)
    sequences, results = data_processor.generate_sequences()

    # --- Initialize Neo4j ---
    graph_db = RouletteGraphDB(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # --- Initialize Embedding Model ---
    embedder = DeepSequenceEmbedder(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        sequence_length=SEQUENCE_LENGTH,
        rnn_units=RNN_UNITS
    )

    # --- Generate and Store Embeddings ---
    similarity_search = SequenceSimilaritySearch(embedder, graph_db)

    # Split data for potential training/validation (if needed)
    train_sequences, test_sequences, train_results, test_results = train_test_split(
        sequences, results, test_size=0.2, random_state=42
    )

    # For demonstration, let's generate and store embeddings for the training sequences
    print("Generating and storing embeddings...")
    train_embeddings = similarity_search.generate_embeddings(data_processor.prepare_for_embedding(train_sequences))
    similarity_search.store_embeddings_neo4j(train_sequences, train_embeddings, train_results)
    print("Embeddings generated and stored in Neo4j.")

    # --- Perform Similarity Search ---
    # Example query
    example_sequence = train_sequences[0]
    print(f"\nFinding sequences similar to: {example_sequence}")
    similar_sequences = similarity_search.find_similar_sequences(example_sequence)
    for seq in similar_sequences:
        print(f"Sequence: {seq['sequence']}, Next Value: {seq['next_value']}, Similarity: {seq['similarity']:.4f}")

    # --- Clean up ---
    graph_db.close()
    print("\nScript finished.")