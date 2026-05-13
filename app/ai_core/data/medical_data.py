"""Seed medical knowledge graph data extracted from the original notebook.

This dataset is intentionally small so the module can run in smoke/offline mode.
Production should seed Neo4j and can extend this data via `training/train.py`.
"""

MEDICAL_DATA: dict[str, list[dict[str, str]]] = {
    'diseases': [
        {'id': 'D001', 'name': 'Đái tháo đường type 2', 'icd': 'E11', 'description': 'Bệnh rối loạn chuyển hóa đường mãn tính'},
        {'id': 'D002', 'name': 'Tăng huyết áp', 'icd': 'I10', 'description': 'Áp lực máu trong động mạch tăng cao bất thường'},
        {'id': 'D003', 'name': 'Viêm phổi', 'icd': 'J18', 'description': 'Nhiễm trùng nhu mô phổi'},
        {'id': 'D004', 'name': 'Nhồi máu cơ tim', 'icd': 'I21', 'description': 'Hoại tử cơ tim do thiếu máu cục bộ'},
        {'id': 'D005', 'name': 'Đột quỵ não', 'icd': 'I64', 'description': 'Tổn thương não do rối loạn tuần hoàn'},
        {'id': 'D006', 'name': 'Suy tim', 'icd': 'I50', 'description': 'Tim không đủ khả năng bơm máu'},
        {'id': 'D007', 'name': 'Viêm dạ dày', 'icd': 'K29', 'description': 'Viêm niêm mạc dạ dày'},
        {'id': 'D008', 'name': 'Hen phế quản', 'icd': 'J45', 'description': 'Bệnh viêm mãn tính đường thở'},
    ],
    'symptoms': [
        {'id': 'S001', 'name': 'Khát nước nhiều', 'description': 'Cảm giác khát liên tục'},
        {'id': 'S002', 'name': 'Tiểu nhiều', 'description': 'Đi tiểu thường xuyên hơn bình thường'},
        {'id': 'S003', 'name': 'Đau đầu', 'description': 'Đau hoặc căng ở vùng đầu'},
        {'id': 'S004', 'name': 'Chóng mặt', 'description': 'Cảm giác quay cuồng'},
        {'id': 'S005', 'name': 'Khó thở', 'description': 'Cảm giác thiếu không khí'},
        {'id': 'S006', 'name': 'Ho', 'description': 'Phản xạ tống xuất dịch từ đường hô hấp'},
        {'id': 'S007', 'name': 'Sốt', 'description': 'Nhiệt độ cơ thể tăng trên 38°C'},
        {'id': 'S008', 'name': 'Đau ngực', 'description': 'Cảm giác đau hoặc tức ngực'},
        {'id': 'S009', 'name': 'Mệt mỏi', 'description': 'Cảm giác kiệt sức'},
        {'id': 'S010', 'name': 'Buồn nôn', 'description': 'Cảm giác muốn nôn'},
        {'id': 'S011', 'name': 'Tê liệt một bên người', 'description': 'Mất khả năng vận động một bên'},
        {'id': 'S012', 'name': 'Phù chân', 'description': 'Sưng phù ở vùng chân'},
        {'id': 'S013', 'name': 'Thở khò khè', 'description': 'Tiếng thở bất thường'},
    ],
    'drugs': [
        {'id': 'DR001', 'name': 'Metformin', 'generic': 'Metformin HCl', 'class': 'Biguanide'},
        {'id': 'DR002', 'name': 'Insulin', 'generic': 'Insulin', 'class': 'Hormone'},
        {'id': 'DR003', 'name': 'Amlodipine', 'generic': 'Amlodipine besylate', 'class': 'Chẹn kênh calci'},
        {'id': 'DR004', 'name': 'Lisinopril', 'generic': 'Lisinopril', 'class': 'Ức chế ACE'},
        {'id': 'DR005', 'name': 'Amoxicillin', 'generic': 'Amoxicillin trihydrate', 'class': 'Kháng sinh Penicillin'},
        {'id': 'DR006', 'name': 'Aspirin', 'generic': 'Acetylsalicylic acid', 'class': 'NSAID'},
        {'id': 'DR007', 'name': 'Atorvastatin', 'generic': 'Atorvastatin calcium', 'class': 'Statin'},
        {'id': 'DR008', 'name': 'Salbutamol', 'generic': 'Albuterol sulfate', 'class': 'Chủ vận beta-2'},
        {'id': 'DR009', 'name': 'Furosemide', 'generic': 'Furosemide', 'class': 'Lợi tiểu quai'},
        {'id': 'DR010', 'name': 'Omeprazole', 'generic': 'Omeprazole', 'class': 'PPI'},
    ],
    'tests': [
        {'id': 'T001', 'name': 'HbA1c', 'description': 'Đường huyết trung bình 3 tháng', 'normal': '< 5.7%'},
        {'id': 'T002', 'name': 'Đường huyết lúc đói', 'description': 'Glucose máu sau 8h nhịn', 'normal': '70-100 mg/dL'},
        {'id': 'T003', 'name': 'Đo huyết áp', 'description': 'Áp lực tâm thu và tâm trương', 'normal': '< 120/80 mmHg'},
        {'id': 'T004', 'name': 'X-quang ngực', 'description': 'Chụp X-quang lồng ngực', 'normal': 'Phổi trong'},
        {'id': 'T005', 'name': 'ECG', 'description': 'Điện tâm đồ', 'normal': 'Nhịp xoang bình thường'},
        {'id': 'T006', 'name': 'Siêu âm tim', 'description': 'Đánh giá chức năng tim', 'normal': 'EF > 55%'},
        {'id': 'T007', 'name': 'CT não', 'description': 'Chụp cắt lớp vi tính não', 'normal': 'Không tổn thương'},
        {'id': 'T008', 'name': 'Đo SpO2', 'description': 'Độ bão hòa oxy máu', 'normal': '95-100%'},
    ],
    'organs': [
        {'id': 'O001', 'name': 'Tim', 'system': 'Tim mạch'},
        {'id': 'O002', 'name': 'Phổi', 'system': 'Hô hấp'},
        {'id': 'O003', 'name': 'Não', 'system': 'Thần kinh'},
        {'id': 'O004', 'name': 'Tụy', 'system': 'Nội tiết'},
        {'id': 'O005', 'name': 'Thận', 'system': 'Tiết niệu'},
        {'id': 'O006', 'name': 'Dạ dày', 'system': 'Tiêu hóa'},
        {'id': 'O007', 'name': 'Mạch máu', 'system': 'Tim mạch'},
    ],
    'risk_factors': [
        {'id': 'RF001', 'name': 'Béo phì', 'description': 'BMI >= 30 kg/m²'},
        {'id': 'RF002', 'name': 'Hút thuốc lá', 'description': 'Sử dụng thuốc lá thường xuyên'},
        {'id': 'RF003', 'name': 'Ít vận động', 'description': 'Lối sống ít hoạt động thể lực'},
        {'id': 'RF004', 'name': 'Tiền sử gia đình', 'description': 'Người thân có bệnh tương tự'},
        {'id': 'RF005', 'name': 'Chế độ ăn nhiều muối', 'description': 'Tiêu thụ >5g muối/ngày'},
        {'id': 'RF006', 'name': 'Stress', 'description': 'Căng thẳng tâm lý kéo dài'},
    ],
    'complications': [
        {'id': 'C001', 'name': 'Bệnh thận đái tháo đường', 'severity': 'Nặng'},
        {'id': 'C002', 'name': 'Bệnh võng mạc đái tháo đường', 'severity': 'Nặng'},
        {'id': 'C003', 'name': 'Đột quỵ não', 'severity': 'Nguy hiểm tính mạng'},
        {'id': 'C004', 'name': 'Nhồi máu cơ tim', 'severity': 'Nguy hiểm tính mạng'},
        {'id': 'C005', 'name': 'Suy tim sung huyết', 'severity': 'Nặng'},
        {'id': 'C007', 'name': 'Hạ đường huyết', 'severity': 'Nguy hiểm'},
    ],
    'treatments': [
        {'id': 'TR001', 'name': 'Thay đổi lối sống', 'type': 'Non-pharmacological'},
        {'id': 'TR002', 'name': 'Liệu pháp insulin', 'type': 'Pharmacological'},
        {'id': 'TR003', 'name': 'Giảm muối trong chế độ ăn', 'type': 'Non-pharmacological'},
        {'id': 'TR004', 'name': 'Tập thể dục 30 phút/ngày', 'type': 'Non-pharmacological'},
        {'id': 'TR005', 'name': 'Kháng sinh đường uống', 'type': 'Pharmacological'},
        {'id': 'TR006', 'name': 'Thở oxy', 'type': 'Supportive'},
        {'id': 'TR007', 'name': 'Phục hồi chức năng', 'type': 'Rehabilitation'},
        {'id': 'TR008', 'name': 'Thuốc chống đông', 'type': 'Pharmacological'},
    ],
    'guidelines': [
        {'id': 'G001', 'name': 'Hướng dẫn điều trị ĐTĐ type 2 - BYT 2020', 'source': 'Bộ Y tế Việt Nam'},
        {'id': 'G002', 'name': 'Hướng dẫn điều trị THA - ESH/ESC 2023', 'source': 'ESH/ESC'},
        {'id': 'G003', 'name': 'Phác đồ điều trị viêm phổi cộng đồng', 'source': 'Bộ Y tế Việt Nam'},
        {'id': 'G004', 'name': 'Hướng dẫn xử trí đột quỵ cấp', 'source': 'Hội Thần kinh học VN'},
    ],
}

RELATIONSHIPS: list[tuple[str, str, str]] = [
    ('D001', 'HAS_SYMPTOM', 'S001'), ('D001', 'HAS_SYMPTOM', 'S002'), ('D001', 'HAS_SYMPTOM', 'S009'),
    ('D002', 'HAS_SYMPTOM', 'S003'), ('D002', 'HAS_SYMPTOM', 'S004'), ('D002', 'HAS_SYMPTOM', 'S008'),
    ('D003', 'HAS_SYMPTOM', 'S005'), ('D003', 'HAS_SYMPTOM', 'S006'), ('D003', 'HAS_SYMPTOM', 'S007'),
    ('D004', 'HAS_SYMPTOM', 'S008'), ('D004', 'HAS_SYMPTOM', 'S009'), ('D004', 'HAS_SYMPTOM', 'S010'),
    ('D005', 'HAS_SYMPTOM', 'S011'), ('D005', 'HAS_SYMPTOM', 'S003'), ('D005', 'HAS_SYMPTOM', 'S004'),
    ('D006', 'HAS_SYMPTOM', 'S005'), ('D006', 'HAS_SYMPTOM', 'S012'), ('D006', 'HAS_SYMPTOM', 'S009'),
    ('D007', 'HAS_SYMPTOM', 'S010'), ('D007', 'HAS_SYMPTOM', 'S003'),
    ('D008', 'HAS_SYMPTOM', 'S005'), ('D008', 'HAS_SYMPTOM', 'S013'), ('D008', 'HAS_SYMPTOM', 'S006'),
    ('D001', 'TREATED_BY', 'DR001'), ('D001', 'TREATED_BY', 'DR002'),
    ('D002', 'TREATED_BY', 'DR003'), ('D002', 'TREATED_BY', 'DR004'),
    ('D003', 'TREATED_BY', 'DR005'),
    ('D004', 'TREATED_BY', 'DR006'), ('D004', 'TREATED_BY', 'DR007'),
    ('D006', 'TREATED_BY', 'DR009'), ('D007', 'TREATED_BY', 'DR010'), ('D008', 'TREATED_BY', 'DR008'),
    ('D001', 'DIAGNOSED_BY', 'T001'), ('D001', 'DIAGNOSED_BY', 'T002'),
    ('D002', 'DIAGNOSED_BY', 'T003'),
    ('D003', 'DIAGNOSED_BY', 'T004'), ('D003', 'DIAGNOSED_BY', 'T008'),
    ('D004', 'DIAGNOSED_BY', 'T005'), ('D004', 'DIAGNOSED_BY', 'T006'),
    ('D005', 'DIAGNOSED_BY', 'T007'),
    ('D001', 'AFFECTS', 'O004'), ('D001', 'AFFECTS', 'O005'),
    ('D002', 'AFFECTS', 'O007'), ('D002', 'AFFECTS', 'O001'),
    ('D003', 'AFFECTS', 'O002'), ('D004', 'AFFECTS', 'O001'),
    ('D005', 'AFFECTS', 'O003'), ('D006', 'AFFECTS', 'O001'), ('D007', 'AFFECTS', 'O006'),
    ('RF001', 'INCREASES_RISK_OF', 'D001'), ('RF003', 'INCREASES_RISK_OF', 'D001'),
    ('RF002', 'INCREASES_RISK_OF', 'D002'), ('RF005', 'INCREASES_RISK_OF', 'D002'),
    ('RF004', 'INCREASES_RISK_OF', 'D001'), ('RF004', 'INCREASES_RISK_OF', 'D002'),
    ('RF001', 'INCREASES_RISK_OF', 'D002'), ('RF006', 'INCREASES_RISK_OF', 'D002'),
    ('D001', 'CAN_CAUSE', 'C001'), ('D001', 'CAN_CAUSE', 'C002'), ('D001', 'CAN_CAUSE', 'C007'),
    ('D002', 'CAN_CAUSE', 'C003'), ('D002', 'CAN_CAUSE', 'C004'), ('D002', 'CAN_CAUSE', 'C005'),
    ('D001', 'MANAGED_BY', 'TR001'), ('D001', 'MANAGED_BY', 'TR002'),
    ('D002', 'MANAGED_BY', 'TR003'), ('D002', 'MANAGED_BY', 'TR004'),
    ('D003', 'MANAGED_BY', 'TR005'), ('D003', 'MANAGED_BY', 'TR006'),
    ('D005', 'MANAGED_BY', 'TR007'), ('D005', 'MANAGED_BY', 'TR008'),
    ('D001', 'FOLLOWS', 'G001'), ('D002', 'FOLLOWS', 'G002'),
    ('D003', 'FOLLOWS', 'G003'), ('D005', 'FOLLOWS', 'G004'),
]


def get_node_type(node_id: str) -> str:
    if node_id.startswith('DR'):
        return 'Drug'
    if node_id.startswith('RF'):
        return 'RiskFactor'
    if node_id.startswith('TR'):
        return 'Treatment'
    return {
        'D': 'Disease',
        'S': 'Symptom',
        'T': 'Test',
        'O': 'Organ',
        'C': 'Complication',
        'G': 'Guideline',
    }.get(node_id[0], 'Unknown')


def iter_all_nodes() -> list[tuple[str, dict[str, str]]]:
    out: list[tuple[str, dict[str, str]]] = []
    key_to_label = {
        'diseases': 'Disease',
        'symptoms': 'Symptom',
        'drugs': 'Drug',
        'tests': 'Test',
        'organs': 'Organ',
        'risk_factors': 'RiskFactor',
        'complications': 'Complication',
        'treatments': 'Treatment',
        'guidelines': 'Guideline',
    }
    for key, label in key_to_label.items():
        for item in MEDICAL_DATA[key]:
            out.append((label, item))
    return out
