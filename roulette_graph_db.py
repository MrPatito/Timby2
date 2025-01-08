from neo4j import GraphDatabase
import numpy as np
from datetime import datetime

class RouletteGraphDB:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.initialize_db()

    def initialize_db(self):
        with self.driver.session() as session:
            # Crear índices y constraints
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Number) REQUIRE n.value IS UNIQUE")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Number) ON (n.value)")

    def add_sequence(self, sequence, result, timestamp=None):
        """
        Añade una secuencia y su resultado a la base de datos
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        with self.driver.session() as session:
            session.execute_write(self._create_sequence, sequence, result, timestamp)

    def _create_sequence(self, tx, sequence, result, timestamp):
        # Crear relaciones entre números consecutivos
        query = """
        UNWIND range(0, size($sequence)-2) as i
        MATCH (n1:Number {value: $sequence[i]})
        MATCH (n2:Number {value: $sequence[i+1]})
        MERGE (n1)-[r:FOLLOWED_BY]->(n2)
        ON CREATE SET r.count = 1, r.timestamps = [$timestamp]
        ON MATCH SET r.count = r.count + 1, 
                     r.timestamps = r.timestamps + $timestamp
        """
        tx.run(query, sequence=sequence, timestamp=timestamp)

        # Almacenar el resultado
        query = """
        MATCH (n:Number {value: $last_number})
        MERGE (n)-[r:RESULTED_IN]->(result:Result {
            number: $result,
            color: $color,
            docena: $docena,
            fila: $fila,
            paridad: $paridad,
            alto_bajo: $alto_bajo
        })
        ON CREATE SET r.count = 1
        ON MATCH SET r.count = r.count + 1
        """
        tx.run(query, 
               last_number=sequence[-1],
               result=result['number'],
               color=result['color'],
               docena=result['docena'],
               fila=result['fila'],
               paridad=result['paridad'],
               alto_bajo=result['alto_bajo'])

    def get_pattern_probabilities(self, sequence):
        """
        Obtiene probabilidades basadas en patrones históricos
        """
        with self.driver.session() as session:
            return session.execute_read(self._get_probabilities, sequence)

    def _get_probabilities(self, tx, sequence):
        query = """
        MATCH (n:Number {value: $last_number})-[r:RESULTED_IN]->(result:Result)
        WITH result, count(*) as freq
        RETURN result.color as color,
               result.docena as docena,
               result.fila as fila,
               result.paridad as paridad,
               result.alto_bajo as alto_bajo,
               freq as frequency
        ORDER BY freq DESC
        """
        results = tx.run(query, last_number=sequence[-1])
        
        probabilities = {
            'color': {},
            'docena': {},
            'fila': {},
            'paridad': {},
            'alto_bajo': {}
        }

        total = 0
        for record in results:
            total += record['frequency']
            for key in probabilities.keys():
                value = record[key]
                if value not in probabilities[key]:
                    probabilities[key][value] = 0
                probabilities[key][value] += record['frequency']

        # Normalizar probabilidades
        for key in probabilities.keys():
            for value in probabilities[key]:
                probabilities[key][value] /= total

        return probabilities 