import numpy as np
from qiskit.quantum_info import Operator

# --- 행렬 분해 함수들  ---
def decompose_A_matrix(m: int) -> dict:
    terms = {'I' * m: 2.0}
    if m > 0:
        terms['I' * (m - 1) + '+'] = -1.0
        terms['I' * (m - 1) + '-'] = -1.0
    for i in range(m - 1):
        terms['I' * i + '-' + '+' * (m - 1 - i)] = -1.0
        terms['I' * i + '+' + '-' * (m - 1 - i)] = -1.0
    return terms

def decompose_B_matrix(m: int) -> dict:
    terms = {}
    if m == 1:
        terms.update({'I': 6.0, '+': -4.0, '-': -4.0})
        return terms
    prev_terms = decompose_B_matrix(m - 1)
    for term_str, coeff in prev_terms.items():
        terms['I' + term_str] = coeff
    term2_base = '-' + '+' * (m - 2)
    terms[term2_base + 'I'] = terms.get(term2_base + 'I', 0) + 1.0
    terms[term2_base + '+'] = terms.get(term2_base + '+', 0) - 4.0
    term3_base = '+' + '-' * (m - 2)
    terms[term3_base + 'I'] = terms.get(term3_base + 'I', 0) + 1.0
    terms[term3_base + '-'] = terms.get(term3_base + '-', 0) - 4.0
    return terms

def decompose_C_matrix(m: int) -> dict:
    return {'0' * m: 1.0, '1' * m: 1.0}

# --- Operator 변환 함수 ---
def dict_to_operator(op_dict: dict, num_qubits: int) -> Operator:
    """파울리 문자열 딕셔너리를 Qiskit Operator 객체로 변환"""
    op_map = {
        'I': Operator(np.identity(2, dtype=complex)),
        '+': Operator(np.array([[0, 1], [0, 0]], dtype=complex)),
        '-': Operator(np.array([[0, 0], [1, 0]], dtype=complex)),
        '0': Operator(np.array([[1, 0], [0, 0]], dtype=complex)),
        '1': Operator(np.array([[0, 0], [0, 1]], dtype=complex))
    }
    
    total_op = Operator(np.zeros((2**num_qubits, 2**num_qubits), dtype=complex))
    
    for term_str, coeff in op_dict.items():
        # 문자열의 첫 글자로 1-큐빗 연산자 초기화
        term_op = op_map[term_str[0]]
        
        # 나머지 문자열에 대해 텐서 곱을 순차적으로 수행
        for char in term_str[1:]:
            term_op = term_op.tensor(op_map[char])
            
        # 계수를 곱하여 전체 연산자에 더해줌
        total_op += coeff * term_op
        
    return total_op