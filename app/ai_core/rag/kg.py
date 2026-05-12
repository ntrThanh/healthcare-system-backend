from __future__ import annotations

import logging
from typing import Any

from app.ai_core.core.config import Settings
from app.ai_core.data.medical_data import MEDICAL_DATA, RELATIONSHIPS, get_node_type

logger = logging.getLogger(__name__)


class MedicalKG:
    """Neo4j Knowledge Graph wrapper."""

    LABEL_CONFIGS = [
        ('diseases', 'Disease'),
        ('symptoms', 'Symptom'),
        ('drugs', 'Drug'),
        ('tests', 'Test'),
        ('organs', 'Organ'),
        ('risk_factors', 'RiskFactor'),
        ('complications', 'Complication'),
        ('treatments', 'Treatment'),
        ('guidelines', 'Guideline'),
    ]

    def __init__(self, uri: str, user: str, password: str, database: str | None = None):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database or None

    @classmethod
    def from_settings(cls, settings: Settings) -> 'MedicalKG | None':
        if not settings.has_neo4j:
            logger.warning('Neo4j credentials are not configured. Falling back to local seed graph.')
            return None
        return cls(
            settings.neo4j_uri or '',
            settings.neo4j_user or '',
            settings.neo4j_password or '',
            settings.neo4j_database,
        )

    def session(self):
        return self.driver.session(database=self.database) if self.database else self.driver.session()

    def create_constraints(self) -> None:
        labels = ['Disease', 'Symptom', 'Drug', 'Test', 'Organ', 'RiskFactor', 'Complication', 'Treatment', 'Guideline']
        with self.session() as session:
            for label in labels:
                session.run(f'CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE')

    def import_seed_data(self) -> None:
        self.create_constraints()
        with self.session() as session:
            for key, label in self.LABEL_CONFIGS:
                for item in MEDICAL_DATA[key]:
                    props = ', '.join([f'n.{k}=${k}' for k in item if k != 'id'])
                    session.run(f'MERGE (n:{label} {{id:$id}}) SET {props}', **item)
                logger.info('Imported %s: %d nodes', label, len(MEDICAL_DATA[key]))

            all_nodes = {item['id']: get_node_type(item['id']) for key in MEDICAL_DATA for item in MEDICAL_DATA[key]}
            count = 0
            for src, rel, tgt in RELATIONSHIPS:
                sl, tl = all_nodes.get(src), all_nodes.get(tgt)
                if not sl or not tl:
                    continue
                session.run(
                    f'MATCH (a:{sl} {{id:$src}}) MATCH (b:{tl} {{id:$tgt}}) MERGE (a)-[:{rel}]->(b)',
                    src=src,
                    tgt=tgt,
                )
                count += 1
            logger.info('Imported relationships: %d', count)

    def stats(self) -> list[dict[str, Any]]:
        with self.session() as session:
            return [dict(r) for r in session.run('MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC')]

    def close(self) -> None:
        self.driver.close()
