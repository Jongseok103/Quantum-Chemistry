# To_Hermite_Operator

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Operator # type: ignore
from Utils.decomposition import *

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


# 아직 성능 최적화가 필요
def get_hermitian_observable(op_str: str) -> SparsePauliOp:
    """비-에르미트 연산자 P에 대해 측정 가능한 에르미트 관측량 O를 생성"""
    num_qubits = len(op_str) # ex) '++-' -> num_qubits = 3
    P = string_to_op(op_str) # Qiskit Operator 객체로 변환
    P_dag = P.adjoint() # 에르미트 켤레 연산자
    Re_P = 0.5 * (P + P_dag) # 실수 부분
    Im_P = -0.5j * (P - P_dag) # 허수 부분
    
    X_op = SparsePauliOp('X') # Pauli X 연산자
    Y_op = SparsePauliOp('Y') # Pauli Y 연산자
    
    # Re(P)나 Im(P)가 0 행렬인 경우를 안정적으로 처리.
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

# 성능 최적화의 대안

def embed_to_hermitian(op_str: str, rtol=1e-5, atol=1e-8) -> SparsePauliOp:
    """
    비-에르미트 연산자를 확장된 힐베르트 공간에서 에르미트 연산자로 임베딩
    
    Args:
        op_str: 연산자 문자열 (예: '+-Z')
        rtol, atol: 0 체크용 허용 오차
        
    Returns:
        (n+1)-큐빗 공간에서의 에르미트 연산자
    """
    num_qubits = len(op_str)
    
    try:
        P = string_to_op(op_str)
        P_dag = P.adjoint()
        
        # 실수부와 허수부 계산
        Re_P = 0.5 * (P + P_dag)
        Im_P = -0.5j * (P - P_dag)
        
        # Pauli 연산자들
        X_op = SparsePauliOp('X')
        Y_op = SparsePauliOp('Y')
        I_extended = SparsePauliOp('I' * (num_qubits + 1))
        
        # 효율적인 0 체크
        def is_negligible(operator):
            if hasattr(operator, 'coeffs'):
                return np.allclose(operator.coeffs, 0, rtol=rtol, atol=atol)
            return np.allclose(operator.data, 0, rtol=rtol, atol=atol)
        
        # 첫 번째 항: X ⊗ Re(P)
        if is_negligible(Re_P):
            term1 = SparsePauliOp('I' * (num_qubits + 1), coeffs= [0])
        else:
            Re_P_pauli = SparsePauliOp.from_operator(Re_P)
            term1 = X_op.tensor(Re_P_pauli)
        
        # 두 번째 항: Y ⊗ Im(P)  
        if is_negligible(Im_P):
            term2 = SparsePauliOp('I' * (num_qubits + 1), coeffs=[0])
        else:
            Im_P_pauli = SparsePauliOp.from_operator(Im_P)
            term2 = Y_op.tensor(Im_P_pauli)
        
        result = term1 + term2
        
        # 결과 검증
        if not result.is_hermitian():
            raise ValueError("결과가 에르미트가 아닙니다")
            
        return result
        
    except Exception as e:
        raise RuntimeError(f"연산자 변환 실패: {e}")
