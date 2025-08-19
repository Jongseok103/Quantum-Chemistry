# To_Hermite_Operator

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Operator # type: ignore
from decomposition import *

# 다른 파일에서 필요한 함수들을 임포트합니다.
from decomposition import decompose_B_matrix
from Ansatz import create_qaoa_ansatz

# --- 헬퍼 함수 (이전 코드에서 가져옴) ---

def string_to_op(op_str: str) -> Operator:
    """I, +, - 문자열을 Qiskit Operator 객체로 변환합니다."""
    X = Operator.from_label('X')
    Y = Operator.from_label('Y')
    I = Operator.from_label('I')
    op_map = {
        'I': I,
        '+': 0.5 * (X + 1j * Y),
        '-': 0.5 * (X - 1j * Y)
    }
    num_qubits = len(op_str)
    if num_qubits == 0: return Operator(np.array([[1]]))
    total_op = op_map[op_str[0]]
    for char in op_str[1:]:
        total_op = total_op.tensor(op_map[char])
    return total_op

def get_hermitian_observable(op_str: str) -> SparsePauliOp:
    """비-에르미트 연산자 P에 대해 측정 가능한 에르미트 관측량 O를 생성합니다."""
    num_qubits = len(op_str)
    P = string_to_op(op_str)
    P_dag = P.adjoint()
    Re_P = 0.5 * (P + P_dag)
    Im_P = -0.5j * (P - P_dag)
    
    X_op = SparsePauliOp('X')
    Y_op = SparsePauliOp('Y')
    
    # Re(P)나 Im(P)가 0 행렬인 경우를 안정적으로 처리합니다.
    if np.allclose(Re_P.to_matrix(), np.zeros_like(Re_P.to_matrix())):
        term1 = SparsePauliOp('I' * (num_qubits + 1), coeffs=[0])
    else:
        Re_P_pauli = SparsePauliOp.from_operator(Re_P)
        term1 = X_op.tensor(Re_P_pauli)

    if np.allclose(Im_P.to_matrix(), np.zeros_like(Im_P.to_matrix())):
        term2 = SparsePauliOp('I' * (num_qubits + 1), coeffs=[0])
    else:
        Im_P_pauli = SparsePauliOp.from_operator(Im_P)
        term2 = Y_op.tensor(Im_P_pauli)
        
    return term1 + term2