import os
import warnings
import numpy as np
from tensorflow.keras.models import Model
import tensorflow as tf
from datetime import datetime
import pandas as pd
import time
from neo4j import GraphDatabase
import concurrent.futures

# Suprimir advertencias de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Desactivar operaciones oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class RouletteEmbedding:
    def __init__(self, model, csv_path, encoders, sequence_length=40, overlap_factor=0.5, batch_size=32):
        """
        Args:
            sequence_length: Length of sequences to process
            overlap_factor: How much sequences should overlap (0.0-1.0)
            batch_size: Size of batches for processing
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.overlap_step = int(sequence_length * (1 - overlap_factor))
        
        print("\n=== Iniciando Sistema de Embeddings Mejorado ===")
        start_time = time.time()
        
        print("1. Cargando modelo...")
        self.model = model
        self.encoders = encoders
        
        print("2. Cargando datos históricos...")
        self.load_historical_data(csv_path)
        
        print("3. Configurando modelo de embedding...")
        self.embedding_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('dense_1').output
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Inicialización completada en {elapsed_time:.2f} segundos")
    
    def load_historical_data(self, csv_path):
        """Carga los datos históricos del CSV"""
        try:
            self.data = pd.read_csv(csv_path)
            self.number_data = {
                row['Num']: {
                    'Color': row['Color'],
                    'Docena': row['Docena'],
                    'Fila': row['Fila'],
                    'Par_Non': row['Par_Non'],
                    'Mayor_Menor': row['Mayor_Menor']
                }
                for _, row in self.data.iterrows()
            }
            print(f"✓ Datos históricos cargados: {len(self.data)} registros")
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            raise

    def get_sequence_embedding(self, sequence, use_augmentation=True):
        """Genera embedding con aumentación de datos opcional"""
        features = self._prepare_sequence_features(sequence)
        if features is None:
            return None
            
        if use_augmentation:
            # Generate variations of the sequence for more robust embedding
            augmented_features = []
            # Original sequence
            augmented_features.append(features)
            
            # Sliding window variations
            for i in range(1, min(5, len(sequence))):
                shifted = np.roll(features, i, axis=1)
                augmented_features.append(shifted)
            
            # Average all embeddings
            embeddings = []
            for feat in augmented_features:
                emb = self.embedding_model.predict(feat, verbose=0)
                embeddings.append(emb)
            
            return np.mean(embeddings, axis=0)
        
        return self.embedding_model.predict(features, verbose=0)

    def _prepare_sequence_features(self, sequence):
        """Prepara las características de la secuencia para el modelo"""
        sequence_features = []
        for numero in sequence:
            features = self._get_features_for_number(numero)
            if features is not None:
                sequence_features.append(features)
            else:
                print(f"Advertencia: No se pudieron obtener características para el número {numero}")
                return None
        return np.array([sequence_features])

    def _get_features_for_number(self, num):
        """Obtiene las características codificadas para un número"""
        try:
            if num not in self.number_data:
                print(f"Error: Número {num} no encontrado en los datos históricos")
                return None
                
            data = self.number_data[num]
            
            color_feat = self.encoders['color'].transform([[data['Color']]])
            docena_feat = self.encoders['docena'].transform([[data['Docena']]])
            fila_feat = self.encoders['fila'].transform([[data['Fila']]])
            paridad_feat = self.encoders['paridad'].transform([[data['Par_Non']]])
            alto_bajo_feat = self.encoders['alto_bajo'].transform([[data['Mayor_Menor']]])

            return np.concatenate([
                color_feat[0], docena_feat[0], fila_feat[0], 
                paridad_feat[0], alto_bajo_feat[0]
            ])
        except Exception as e:
            print(f"Error extracting features for number {num}: {e}")
            return None

    def get_historical_sequence(self, length=40):
        """Obtiene la última secuencia histórica del CSV"""
        return self.data['Num'].tail(length).tolist()

class SequenceDatabase:
    def __init__(self, graph_db, embedding_system, batch_size=100, max_workers=4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        print("\n=== Iniciando Base de Datos de Secuencias ===")
        self.graph_db = graph_db
        self.embedding_system = embedding_system
        self.historical_sequences = []
        self.initialize_from_historical_data()

    def initialize_from_historical_data(self):
        """Inicializa la base de datos con procesamiento por lotes"""
        print("1. Procesando secuencias históricas...")
        start_time = time.time()
        data = self.embedding_system.data
        window_size = self.embedding_system.sequence_length
        step_size = self.embedding_system.overlap_step
        
        total_sequences = (len(data) - window_size) // step_size
        
        print(f"- Total de secuencias a procesar: {total_sequences}")
        print(f"- Tamaño de lote: {self.batch_size}")
        print(f"- Trabajadores paralelos: {self.max_workers}")
        
        # Process in batches
        for batch_start in range(0, total_sequences, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_sequences)
            batch_sequences = []
            
            for i in range(batch_start, batch_end):
                idx = i * step_size
                sequence = data['Num'].iloc[idx:idx+window_size].tolist()
                result = data['Num'].iloc[idx+window_size]
                batch_sequences.append((sequence, result))
            
            # Process batch in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                executor.map(lambda x: self.store_sequence(*x), batch_sequences)
            
            print(f"- Progreso: {batch_end}/{total_sequences} secuencias procesadas")
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Base de datos inicializada en {elapsed_time:.2f} segundos")
        print(f"✓ Total de secuencias almacenadas: {len(self.historical_sequences)}")

    def store_sequence(self, sequence, result):
        """Almacena la secuencia con su embedding"""
        embedding = self.embedding_system.get_sequence_embedding(sequence)
        if embedding is not None:
            timestamp = datetime.now().timestamp()
            
            with self.graph_db.driver.session() as session:
                session.execute_write(
                    self._store_sequence_transaction,
                    sequence,
                    result,
                    embedding.tolist(),
                    timestamp
                )
            
            self.historical_sequences.append({
                'sequence': sequence,
                'result': result,
                'timestamp': timestamp
            })

    def _store_sequence_transaction(self, tx, sequence, result, embedding, timestamp):
        # Flatten the embedding array to a 1D list
        flattened_embedding = [float(x) for x in embedding[0]]  # Extract and flatten first sequence
        
        # Create node of sequence with flattened embedding
        query = """
        CREATE (s:Sequence {
            timestamp: $timestamp,
            numbers: $sequence,
            embedding: $embedding
        })
        WITH s
        MATCH (n:Number {value: $result_number})
        CREATE (s)-[:RESULTED_IN]->(n)
        """
        tx.run(query, 
               timestamp=timestamp,
               sequence=sequence,
               embedding=flattened_embedding,  # Now using flattened embedding
               result_number=result)

    def find_similar_sequences(self, current_sequence, limit=5):
        """
        Encuentra secuencias similares basadas en embeddings
        """
        current_embedding = self.embedding_system.get_sequence_embedding(current_sequence)
        if current_embedding is None:
            return []
        
        # Flatten the embedding to match stored format
        flattened_embedding = [float(x) for x in current_embedding[0]]
        
        with self.graph_db.driver.session() as session:
            return session.execute_read(
                self._find_similar_sequences_transaction,
                flattened_embedding,
                limit
            )

    def _find_similar_sequences_transaction(self, tx, embedding, limit):
        """
        Encuentra secuencias similares usando similitud coseno implementada en Cypher
        """
        query = """
        WITH $embedding AS target
        MATCH (s:Sequence)-[:RESULTED_IN]->(r:Number)
        WITH s, r, 
             reduce(dot = 0.0, i in range(0, size(s.embedding)-1) | 
               dot + s.embedding[i] * target[i]) /
             (sqrt(reduce(l2 = 0.0, i in range(0, size(s.embedding)-1) | 
               l2 + s.embedding[i] * s.embedding[i])) *
              sqrt(reduce(l2 = 0.0, i in range(0, size(target)-1) | 
               l2 + target[i] * target[i]))) AS similarity
        WHERE similarity IS NOT NULL
        ORDER BY similarity DESC
        LIMIT $limit
        RETURN s.numbers AS sequence, 
               r.value AS result, 
               similarity,
               s.timestamp AS timestamp
        """
        
        results = list(tx.run(query, 
                             embedding=embedding,
                             limit=limit))
        
        # Format results for better readability
        formatted_results = [{
            'sequence': result['sequence'],
            'result': result['result'],
            'similarity': round(result['similarity'], 4),
            'timestamp': result['timestamp']
        } for result in results]
        
        return formatted_results

    def validate_database_utility(self, test_sequences=None, verbose=True):
        """
        Valida la utilidad de la base de datos analizando secuencias de prueba
        
        Args:
            test_sequences: Lista de secuencias para probar. Si es None, usa las últimas del histórico
            verbose: Si es True, muestra información detallada
        """
        if test_sequences is None:
            # Usar las últimas 5 secuencias del histórico
            test_sequences = self.historical_sequences[-5:]
        
        results = []
        for test_case in test_sequences:
            sequence = test_case['sequence']
            actual_result = test_case['result']
            
            # Encontrar secuencias similares
            similar_sequences = self.find_similar_sequences(sequence, limit=10)
            
            if not similar_sequences:
                if verbose:
                    print(f"⚠️ No se encontraron secuencias similares para: [...{sequence[-5:]}]")
                continue
            
            # Analizar resultados
            predictions = [seq['result'] for seq in similar_sequences]
            similarities = [seq['similarity'] for seq in similar_sequences]
            
            # Calcular estadísticas
            avg_similarity = sum(similarities) / len(similarities)
            result_distribution = {}
            for pred, sim in zip(predictions, similarities):
                if pred not in result_distribution:
                    result_distribution[pred] = {'count': 0, 'total_similarity': 0}
                result_distribution[pred]['count'] += 1
                result_distribution[pred]['total_similarity'] += sim
            
            # Ordenar resultados por peso (frecuencia * similitud)
            weighted_results = {
                num: (data['count'] * data['total_similarity']) 
                for num, data in result_distribution.items()
            }
            top_predictions = sorted(
                weighted_results.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            if verbose:
                print("\n=== Análisis de Secuencia ===")
                print(f"Secuencia: [...{sequence[-5:]}]")
                print(f"Resultado real: {actual_result}")
                print("\nTop 3 predicciones más probables:")
                for num, weight in top_predictions:
                    count = result_distribution[num]['count']
                    print(f"• Número {num}: {count} ocurrencias, peso: {weight:.4f}")
                print(f"\nSimilitud promedio: {avg_similarity:.4f}")
                print(f"Secuencias similares encontradas: {len(similar_sequences)}")
            
            results.append({
                'sequence': sequence,
                'actual_result': actual_result,
                'predictions': top_predictions,
                'avg_similarity': avg_similarity,
                'found_sequences': len(similar_sequences)
            })
        
        # Calcular métricas globales
        total_sequences = len(results)
        # Check if total_sequences is zero to avoid division by zero
        if total_sequences == 0:
            if verbose:
                print("⚠️ No se analizaron secuencias. No se puede calcular la similitud promedio global.")
            return results  # Return early if no sequences were analyzed

        sequences_with_matches = len([r for r in results if r['found_sequences'] > 0])
        avg_similarity_global = sum(r['avg_similarity'] for r in results) / total_sequences
        
        if verbose:
            print("\n=== Métricas Globales ===")
            print(f"Total secuencias analizadas: {total_sequences}")
            print(f"Secuencias con coincidencias: {sequences_with_matches}")
            print(f"Tasa de cobertura: {(sequences_with_matches/total_sequences)*100:.2f}%")
            print(f"Similitud promedio global: {avg_similarity_global:.4f}")
        
        return results

class RouletteGraphDB:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="123456789"):
        """Inicializa la conexión a Neo4j"""
        print("\n=== Conectando a Neo4j ===")
        try:
            # Verify Neo4j is accessible
            print(f"Intentando conectar a {uri}")
            self.driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=5)
            self.test_connection()
            print("✓ Conexión exitosa a Neo4j")
            self.initialize_db()
        except Exception as e:
            if "authentication failure" in str(e).lower():
                print(f"❌ Error de autenticación. Verifique usuario y contraseña.")
                print(f"Usuario: {user}")
                print(f"URI: {uri}")
            else:
                print(f"❌ Error conectando a Neo4j: {e}")
            raise

    def test_connection(self):
        """Prueba la conexión a Neo4j"""
        try:
            with self.driver.session() as session:  # Removed timeout parameter
                result = session.run("RETURN 1")
                result.single()[0]
        except Exception as e:
            print("❌ Error en test de conexión:")
            print(f"  - Asegúrese que Neo4j está corriendo")
            print(f"  - Verifique las credenciales")
            print(f"  - Error específico: {str(e)}")
            raise

    def initialize_db(self):
        """Inicializa la estructura de la base de datos"""
        with self.driver.session() as session:
            # Crear constraints e índices
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Number) REQUIRE n.value IS UNIQUE")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Number) ON (n.value)")
            
            # Crear nodos para todos los números de la ruleta
            query = """
            UNWIND range(0, 36) as num
            MERGE (n:Number {value: num})
            """
            session.run(query)
            
            # Verificar la creación
            result = session.run("MATCH (n:Number) RETURN count(n) as count")
            count = result.single()["count"]
            print(f"✓ {count} números inicializados en la base de datos")

    def get_database_stats(self):
        """Obtiene estadísticas de la base de datos"""
        with self.driver.session() as session:
            stats = {}
            
            # Contar nodos de número
            result = session.run("MATCH (n:Number) RETURN count(n) as count")
            stats['numbers'] = result.single()['count']
            
            # Contar nodos de secuencia
            result = session.run("MATCH (s:Sequence) RETURN count(s) as count")
            stats['sequences'] = result.single()['count']
            
            # Contar relaciones
            result = session.run("MATCH ()-[r:RESULTED_IN]->() RETURN count(r) as count")
            stats['relationships'] = result.single()['count']
            
            return stats

# Example of usage
if __name__ == "__main__":
    print("\n🎲 INICIANDO SISTEMA DE EMBEDDINGS DE RULETA 🎲")
    start_total = time.time()
    
    try:
        # 1. Inicializar conexión a Neo4j
        graph_db = RouletteGraphDB(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="123456789"
        )
        
        # 2. Cargar modelo y encoders
        print("\n=== Cargando Recursos ===")
        from tensorflow.keras.models import load_model
        import pickle
        
        print("1. Cargando modelo...")
        modelo = load_model("modelo_ruleta_CMulti_Cursor2.h5")
        print("✓ Modelo cargado correctamente")
        
        print("\n2. Cargando encoders...")
        with open('encoders.pickle', 'rb') as handle:
            encoders = pickle.load(handle)
        print("✓ Encoders cargados correctamente")
        
        # 3. Crear sistema de embeddings
        embedding_system = RouletteEmbedding(
            model=modelo,
            csv_path="secuencia_ruleta_V2.csv",
            encoders=encoders,
            sequence_length=50,  # Longer sequences
            overlap_factor=0.7,  # More overlap
            batch_size=64       # Larger batches
        )
        
        # 4. NUEVO: Inicializar y poblar la base de datos
        print("\n=== Limpiando Base de Datos ===")
        with graph_db.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✓ Base de datos limpiada")
        
        sequence_db = SequenceDatabase(
            graph_db, 
            embedding_system,
            batch_size=200,    # Process more sequences at once
            max_workers=8      # Use more CPU cores
        )
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise 