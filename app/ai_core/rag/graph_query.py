from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Protocol

from app.ai_core.data.medical_data import MEDICAL_DATA, RELATIONSHIPS

logger = logging.getLogger(__name__)


class GraphEngine(Protocol):
    def get_disease_info(self, name: str) -> dict[str, Any] | None: ...
    def get_symptom_diseases(self, name: str) -> list[dict[str, Any]]: ...
    def get_drug_info(self, name: str) -> dict[str, Any] | None: ...
    def get_multihop_context(self, disease_name: str) -> str: ...
    def list_names(self, label: str) -> list[str]: ...


class Neo4jGraphQueryEngine:
    def __init__(self, driver, database: str | None = None):
        self.driver = driver
        self.database = database or None
        self._create_indexes()

    def _session(self):
        return self.driver.session(database=self.database) if self.database else self.driver.session()

    def _create_indexes(self) -> None:
        indexes = [
            'CREATE INDEX disease_name IF NOT EXISTS FOR (n:Disease) ON (n.name)',
            'CREATE INDEX symptom_name IF NOT EXISTS FOR (n:Symptom) ON (n.name)',
            'CREATE INDEX drug_name IF NOT EXISTS FOR (n:Drug) ON (n.name)',
            'CREATE INDEX test_name IF NOT EXISTS FOR (n:Test) ON (n.name)',
        ]
        with self._session() as session:
            for idx in indexes:
                try:
                    session.run(idx)
                except Exception as exc:
                    logger.debug('Index creation skipped: %s', exc)

    def list_names(self, label: str) -> list[str]:
        with self._session() as session:
            return [r['name'] for r in session.run(f'MATCH (n:{label}) RETURN n.name AS name')]

    def get_disease_info(self, name: str) -> dict[str, Any] | None:
        query = '''
        MATCH (d:Disease)
        WHERE toLower(d.name) CONTAINS toLower($name)
        OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
        OPTIONAL MATCH (d)-[:TREATED_BY]->(dr:Drug)
        OPTIONAL MATCH (d)-[:DIAGNOSED_BY]->(t:Test)
        OPTIONAL MATCH (d)-[:AFFECTS]->(o:Organ)
        OPTIONAL MATCH (rf:RiskFactor)-[:INCREASES_RISK_OF]->(d)
        OPTIONAL MATCH (d)-[:CAN_CAUSE]->(c:Complication)
        OPTIONAL MATCH (d)-[:MANAGED_BY]->(tr:Treatment)
        OPTIONAL MATCH (d)-[:FOLLOWS]->(g:Guideline)
        RETURN d.name as disease, d.description as desc, d.icd as icd,
               collect(DISTINCT s.name)[..10] as symptoms,
               collect(DISTINCT dr.name)[..10] as drugs,
               collect(DISTINCT t.name)[..10] as tests,
               collect(DISTINCT o.name)[..5] as organs,
               collect(DISTINCT rf.name)[..5] as risk_factors,
               collect(DISTINCT c.name)[..5] as complications,
               collect(DISTINCT tr.name)[..5] as treatments,
               collect(DISTINCT g.name)[..3] as guidelines
        LIMIT 1'''
        with self._session() as session:
            rows = [dict(r) for r in session.run(query, name=name)]
            return rows[0] if rows else None

    def get_symptom_diseases(self, name: str) -> list[dict[str, Any]]:
        query = '''MATCH (s:Symptom)<-[:HAS_SYMPTOM]-(d:Disease)
                   WHERE toLower(s.name) CONTAINS toLower($name)
                   RETURN d.name as disease, d.description as desc LIMIT 5'''
        with self._session() as session:
            return [dict(r) for r in session.run(query, name=name)]

    def get_drug_info(self, name: str) -> dict[str, Any] | None:
        query = '''MATCH (dr:Drug) WHERE toLower(dr.name) CONTAINS toLower($name)
                   OPTIONAL MATCH (d:Disease)-[:TREATED_BY]->(dr)
                   RETURN dr.name as drug, dr.generic as generic, dr.class as drug_class,
                          collect(DISTINCT d.name)[..5] as treats LIMIT 1'''
        with self._session() as session:
            rows = [dict(r) for r in session.run(query, name=name)]
            return rows[0] if rows else None

    @staticmethod
    def format(info: dict[str, Any] | None) -> str:
        if not info:
            return ''
        clean = lambda values: [x for x in values if x]
        lines: list[str] = []
        if 'disease' in info:
            lines.extend([
                f"Bệnh: {info['disease']} (ICD: {info.get('icd', 'N/A')})",
                f"Mô tả: {info.get('desc', 'N/A')}",
            ])
            for key, label in [
                ('symptoms', 'Triệu chứng'),
                ('drugs', 'Thuốc điều trị'),
                ('tests', 'Xét nghiệm'),
                ('organs', 'Cơ quan ảnh hưởng'),
                ('risk_factors', 'Yếu tố nguy cơ'),
                ('complications', 'Biến chứng'),
                ('treatments', 'Điều trị'),
                ('guidelines', 'Hướng dẫn'),
            ]:
                if clean(info.get(key, [])):
                    lines.append(f"{label}: {', '.join(clean(info[key]))}")
        elif 'drug' in info:
            lines.extend([f"Thuốc: {info['drug']} ({info.get('generic', '')})", f"Nhóm: {info.get('drug_class', '')}"])
            if clean(info.get('treats', [])):
                lines.append(f"Điều trị: {', '.join(clean(info['treats']))}")
        return '\n'.join(lines)

    MULTI_HOP_QUERY = '''
    MATCH (d:Disease)
    WHERE toLower(d.name) CONTAINS toLower($name)
    WITH d LIMIT 1
    MATCH path = (d)-[:HAS_SYMPTOM|TREATED_BY|DIAGNOSED_BY|FOLLOWS|CAN_CAUSE|MANAGED_BY|AFFECTS*1..2]->(related)
    WITH d, labels(related)[0] AS related_type, related.name AS related_name, related.description AS related_desc
    RETURN d.name AS disease, related_type, related_name, related_desc
    LIMIT 30
    '''

    @lru_cache(maxsize=128)
    def get_multihop_context(self, disease_name: str) -> str:
        with self._session() as session:
            rows = list(session.run(self.MULTI_HOP_QUERY, name=disease_name))
        if not rows:
            return self.format(self.get_disease_info(disease_name))
        sections: dict[str, list[str]] = {}
        for row in rows:
            rtype = row['related_type'] or 'Unknown'
            rname = row['related_name'] or ''
            rdesc = row['related_desc'] or ''
            entry = rname + (f' ({rdesc[:60]})' if rdesc and rdesc != rname else '')
            sections.setdefault(rtype, [])
            if entry not in sections[rtype]:
                sections[rtype].append(entry)
        return _format_multihop(disease_name, sections)


class LocalGraphQueryEngine:
    """In-memory graph engine for offline smoke tests and fallback mode."""

    def __init__(self):
        self.nodes = {item['id']: item for key in MEDICAL_DATA for item in MEDICAL_DATA[key]}
        self.names = {item['name'].lower(): item for item in self.nodes.values()}

    def list_names(self, label: str) -> list[str]:
        key_by_label = {
            'Disease': 'diseases',
            'Symptom': 'symptoms',
            'Drug': 'drugs',
            'Test': 'tests',
        }
        key = key_by_label.get(label)
        return [item['name'] for item in MEDICAL_DATA.get(key, [])] if key else []

    def _find_by_name(self, group: str, name: str) -> dict[str, Any] | None:
        q = name.lower()
        for item in MEDICAL_DATA[group]:
            item_name = item['name'].lower()
            if q in item_name or any(tok in q for tok in item_name.split()[:2]):
                return item
        return None

    def _targets(self, source_id: str, rel: str) -> list[dict[str, Any]]:
        return [self.nodes[tgt] for src, r, tgt in RELATIONSHIPS if src == source_id and r == rel and tgt in self.nodes]

    def _sources(self, target_id: str, rel: str) -> list[dict[str, Any]]:
        return [self.nodes[src] for src, r, tgt in RELATIONSHIPS if tgt == target_id and r == rel and src in self.nodes]

    def get_disease_info(self, name: str) -> dict[str, Any] | None:
        d = self._find_by_name('diseases', name)
        if not d:
            return None
        return {
            'disease': d['name'],
            'desc': d.get('description', ''),
            'icd': d.get('icd', ''),
            'symptoms': [x['name'] for x in self._targets(d['id'], 'HAS_SYMPTOM')],
            'drugs': [x['name'] for x in self._targets(d['id'], 'TREATED_BY')],
            'tests': [x['name'] for x in self._targets(d['id'], 'DIAGNOSED_BY')],
            'organs': [x['name'] for x in self._targets(d['id'], 'AFFECTS')],
            'risk_factors': [x['name'] for x in self._sources(d['id'], 'INCREASES_RISK_OF')],
            'complications': [x['name'] for x in self._targets(d['id'], 'CAN_CAUSE')],
            'treatments': [x['name'] for x in self._targets(d['id'], 'MANAGED_BY')],
            'guidelines': [x['name'] for x in self._targets(d['id'], 'FOLLOWS')],
        }

    def get_symptom_diseases(self, name: str) -> list[dict[str, Any]]:
        symptom = self._find_by_name('symptoms', name)
        if not symptom:
            return []
        return [{'disease': d['name'], 'desc': d.get('description', '')} for d in self._sources(symptom['id'], 'HAS_SYMPTOM')]

    def get_drug_info(self, name: str) -> dict[str, Any] | None:
        drug = self._find_by_name('drugs', name)
        if not drug:
            return None
        treats = self._sources(drug['id'], 'TREATED_BY')
        return {
            'drug': drug['name'],
            'generic': drug.get('generic', ''),
            'drug_class': drug.get('class', ''),
            'treats': [d['name'] for d in treats],
        }

    format = staticmethod(Neo4jGraphQueryEngine.format)

    @lru_cache(maxsize=128)
    def get_multihop_context(self, disease_name: str) -> str:
        info = self.get_disease_info(disease_name)
        if not info:
            return ''
        sections = {
            'Symptom': info.get('symptoms', []),
            'Drug': info.get('drugs', []),
            'Test': info.get('tests', []),
            'Organ': info.get('organs', []),
            'RiskFactor': info.get('risk_factors', []),
            'Complication': info.get('complications', []),
            'Treatment': info.get('treatments', []),
            'Guideline': info.get('guidelines', []),
        }
        return _format_multihop(info['disease'], {k: v for k, v in sections.items() if v})


def _format_multihop(disease_name: str, sections: dict[str, list[str]]) -> str:
    type_labels = {
        'Symptom': 'Triệu chứng',
        'Drug': 'Thuốc điều trị',
        'Test': 'Xét nghiệm chẩn đoán',
        'Complication': 'Biến chứng',
        'Treatment': 'Phác đồ điều trị',
        'Guideline': 'Hướng dẫn lâm sàng',
        'Organ': 'Cơ quan ảnh hưởng',
        'RiskFactor': 'Yếu tố nguy cơ',
    }
    lines = [f'[Multi-hop KG] Bệnh: {disease_name}']
    for rtype, items in sections.items():
        label = type_labels.get(rtype, rtype)
        lines.append(f"  {label}: {', '.join(items[:8])}")
    return '\n'.join(lines)
