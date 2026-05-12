"""Feature/corpus loader for RAG.

The notebook built FAISS documents directly from Neo4j. This file turns that into a
reusable production module and also supports local seed data for smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from app.ai_core.data.medical_data import MEDICAL_DATA, RELATIONSHIPS, iter_all_nodes

try:  # pragma: no cover - fallback is for light environments
    from langchain_core.documents import Document as LangChainDocument
except Exception:  # pragma: no cover
    LangChainDocument = None


@dataclass
class SimpleDocument:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _document(content: str, metadata: dict[str, Any]):
    if LangChainDocument is not None:
        return LangChainDocument(page_content=content, metadata=metadata)
    return SimpleDocument(page_content=content, metadata=metadata)


TYPE_CONFIDENCE = {
    'Disease': 0.95,
    'Drug': 0.92,
    'Test': 0.90,
    'Symptom': 0.85,
    'RiskFactor': 0.80,
    'Complication': 0.85,
    'Treatment': 0.88,
    'Guideline': 0.88,
    'QA': 0.90,
}


def make_doc(content: str, meta: dict[str, Any]):
    node_type = meta.get('type', 'Unknown')
    entity = meta.get('name', '')
    enriched = f"[Loại: {node_type}] [Thực thể: {entity}]\n{content}"
    return _document(
        enriched,
        {**meta, 'source_confidence': TYPE_CONFIDENCE.get(node_type, 0.75)},
    )


def _index_seed_data() -> dict[str, dict[str, str]]:
    return {item['id']: item for _, item in iter_all_nodes()}


def _relations_by_source(rel_type: str) -> dict[str, list[str]]:
    nodes = _index_seed_data()
    out: dict[str, list[str]] = {}
    for src, rel, tgt in RELATIONSHIPS:
        if rel == rel_type and src in nodes and tgt in nodes:
            out.setdefault(src, []).append(nodes[tgt]['name'])
    return out


def build_corpus_from_seed_data() -> list[Any]:
    """Build contextual RAG documents from the embedded seed dataset."""
    docs: list[Any] = []

    for d in MEDICAL_DATA['diseases']:
        docs.append(make_doc(
            f"Bệnh {d['name']}: {d.get('description', '')}. ICD: {d.get('icd', '')}",
            {'type': 'Disease', 'id': d['id'], 'name': d['name']},
        ))
    for s in MEDICAL_DATA['symptoms']:
        docs.append(make_doc(
            f"Triệu chứng {s['name']}: {s.get('description', '')}",
            {'type': 'Symptom', 'id': s['id'], 'name': s['name']},
        ))
    for dr in MEDICAL_DATA['drugs']:
        docs.append(make_doc(
            f"Thuốc {dr['name']} ({dr.get('generic', '')}), nhóm {dr.get('class', '')}",
            {'type': 'Drug', 'id': dr['id'], 'name': dr['name']},
        ))
    for t in MEDICAL_DATA['tests']:
        docs.append(make_doc(
            f"Xét nghiệm {t['name']}: {t.get('description', '')}. Bình thường: {t.get('normal', '')}",
            {'type': 'Test', 'id': t['id'], 'name': t['name']},
        ))
    for rf in MEDICAL_DATA['risk_factors']:
        docs.append(make_doc(
            f"Yếu tố nguy cơ {rf['name']}: {rf.get('description', '')}",
            {'type': 'RiskFactor', 'id': rf['id'], 'name': rf['name']},
        ))
    for c in MEDICAL_DATA['complications']:
        docs.append(make_doc(
            f"Biến chứng {c['name']}: mức độ {c.get('severity', '')}",
            {'type': 'Complication', 'id': c['id'], 'name': c['name']},
        ))
    for tr in MEDICAL_DATA['treatments']:
        docs.append(make_doc(
            f"Điều trị {tr['name']}: loại {tr.get('type', '')}",
            {'type': 'Treatment', 'id': tr['id'], 'name': tr['name']},
        ))

    symptoms_by_disease = _relations_by_source('HAS_SYMPTOM')
    drugs_by_disease = _relations_by_source('TREATED_BY')
    tests_by_disease = _relations_by_source('DIAGNOSED_BY')

    for d in MEDICAL_DATA['diseases']:
        if symptoms_by_disease.get(d['id']):
            docs.append(make_doc(
                f"Q: {d['name']} có triệu chứng gì?\nA: {', '.join(symptoms_by_disease[d['id']])}",
                {'type': 'QA', 'id': f"qa_sym_{d['id']}", 'name': d['name']},
            ))
        if drugs_by_disease.get(d['id']):
            docs.append(make_doc(
                f"Q: {d['name']} điều trị bằng thuốc gì?\nA: {', '.join(drugs_by_disease[d['id']])}",
                {'type': 'QA', 'id': f"qa_drug_{d['id']}", 'name': d['name']},
            ))
        if tests_by_disease.get(d['id']):
            docs.append(make_doc(
                f"Q: {d['name']} chẩn đoán/xét nghiệm gì?\nA: {', '.join(tests_by_disease[d['id']])}",
                {'type': 'QA', 'id': f"qa_test_{d['id']}", 'name': d['name']},
            ))

    return docs


def build_corpus_from_neo4j(driver, database: str | None = None) -> list[Any]:
    """Build contextual documents from a Neo4j graph, equivalent to the notebook."""
    docs: list[Any] = []
    session_kwargs = {"database": database} if database else {}
    with driver.session(**session_kwargs) as session:
        for r in session.run('MATCH (d:Disease) RETURN d'):
            d = r['d']
            docs.append(make_doc(
                f"Bệnh {d['name']}: {d.get('description', '')}. ICD: {d.get('icd', '')}",
                {'type': 'Disease', 'id': d['id'], 'name': d['name']},
            ))
        for r in session.run('MATCH (s:Symptom) RETURN s'):
            s = r['s']
            docs.append(make_doc(
                f"Triệu chứng {s['name']}: {s.get('description', '')}",
                {'type': 'Symptom', 'id': s['id'], 'name': s['name']},
            ))
        for r in session.run('MATCH (dr:Drug) RETURN dr'):
            dr = r['dr']
            docs.append(make_doc(
                f"Thuốc {dr['name']} ({dr.get('generic', '')}), nhóm {dr.get('class', '')}",
                {'type': 'Drug', 'id': dr['id'], 'name': dr['name']},
            ))
        for r in session.run('MATCH (t:Test) RETURN t'):
            t = r['t']
            docs.append(make_doc(
                f"Xét nghiệm {t['name']}: {t.get('description', '')}. Bình thường: {t.get('normal', '')}",
                {'type': 'Test', 'id': t['id'], 'name': t['name']},
            ))
        for r in session.run('MATCH (rf:RiskFactor) RETURN rf'):
            rf = r['rf']
            docs.append(make_doc(
                f"Yếu tố nguy cơ {rf['name']}: {rf.get('description', '')}",
                {'type': 'RiskFactor', 'id': rf['id'], 'name': rf['name']},
            ))
        for r in session.run('MATCH (d:Disease)-[:HAS_SYMPTOM]->(sym:Symptom) RETURN d.name AS disease, collect(sym.name) AS symptoms'):
            docs.append(make_doc(
                f"Q: {r['disease']} có triệu chứng gì?\nA: {', '.join(r['symptoms'])}",
                {'type': 'QA', 'id': 'qa_sym', 'name': r['disease']},
            ))
        for r in session.run('MATCH (d:Disease)-[:TREATED_BY]->(dr:Drug) RETURN d.name AS disease, collect(dr.name) AS drugs'):
            docs.append(make_doc(
                f"Q: {r['disease']} điều trị bằng thuốc gì?\nA: {', '.join(r['drugs'])}",
                {'type': 'QA', 'id': 'qa_drug', 'name': r['disease']},
            ))
    return docs


def corpus_to_text(docs: Iterable[Any], max_chars: int = 20_000) -> str:
    text = '\n\n'.join(getattr(d, 'page_content', str(d)) for d in docs)
    return text[:max_chars]
