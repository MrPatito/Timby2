import time
from sequence_embeddings import RouletteGraphDB, RouletteEmbedding, SequenceDatabase, DeepAttentionEmbedding

if __name__ == "__main__":
    print("\n🎲 INICIANDO SISTEMA DE EMBEDDINGS DE RULETA (TEST) 🎲")
    start_total = time.time()

    try:
        # 1. Inicializar conexión a Neo4j
        graph_db = RouletteGraphDB(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="123456789"
        )

        # 2. Configurar el modelo Deep Attention
        print("\n=== Configurando Modelo Deep Attention ===")
        vocab_size = 37  # Números del 0 al 36
        embedding_dim = 64
        sequence_length = 50
        num_heads = 4
        ff_dim = 128
        dense_units = 256

        deep_attention_model = DeepAttentionEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            sequence_length=sequence_length,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dense_units=dense_units
        )
        print("✓ Modelo Deep Attention inicializado")

        # 3. Cargar encoders
        print("\n=== Cargando Recursos ===")
        import pickle
        print("1. Cargando encoders...")
        with open('encoders.pickle', 'rb') as handle:
            encoders = pickle.load(handle)
        print("✓ Encoders cargados correctamente")

        # 4. Crear sistema de embeddings
        embedding_system = RouletteEmbedding(
            deep_attention_model=deep_attention_model,
            csv_path="secuencia_ruleta_V2.csv",
            encoders=encoders,
            sequence_length=sequence_length,
            overlap_factor=0.7,
            batch_size=64
        )

        # 5. Inicializar la base de datos
        sequence_db = SequenceDatabase(
            graph_db,
            embedding_system,
            batch_size=200,
            max_workers=8
        )

        # 6. Obtener y probar una secuencia
        print("\n=== Probando Sistema ===")
        ultima_secuencia = embedding_system.get_historical_sequence()
        print(f"1. Última secuencia obtenida (últimos 5 números): ...{ultima_secuencia[-5:]}")

        embedding = embedding_system.get_sequence_embedding(ultima_secuencia)
        print(f"2. Embedding generado con forma: {embedding.shape if embedding is not None else 'None'}")

        # 7. Mostrar estadísticas actualizadas
        print("\n=== Estadísticas de la Base de Datos ===")
        stats = graph_db.get_database_stats()
        print(f"• Secuencias almacenadas: {stats['sequences']}")
        print(f"• Relaciones creadas: {stats['relationships']}")

        # 8. Validar la utilidad de la base de datos
        print("\n=== Validando Utilidad de la Base de Datos ===")
        test_sequences = sequence_db.historical_sequences[-5:]  # últimas 5 secuencias
        validation_results = sequence_db.validate_database_utility(test_sequences)

        # 9. Mostrar tiempo total
        tiempo_total = time.time() - start_total
        print(f"\n✅ SISTEMA INICIALIZADO Y PROBADO CORRECTAMENTE")
        print(f"⏱️  Tiempo total de ejecución: {tiempo_total:.2f} segundos")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise 