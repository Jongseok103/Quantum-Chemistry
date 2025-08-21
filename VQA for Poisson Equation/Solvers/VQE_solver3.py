# file: VQE_solver.py

import numpy as np
from scipy.optimize import minimize # type: ignore
from typing import TYPE_CHECKING
from qiskit.quantum_info import Statevector

from functools import lru_cache
from typing import Union, Sequence, Dict
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp
from qiskit import QuantumCircuit # type: ignore

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit # type: ignore

# 다른 모듈에서 필요한 클래스 및 함수 임포트
from Utils.decomposition import decompose_A_matrix, decompose_B_matrix, decompose_C_matrix, dict_to_operator
from Utils.create_b_state import get_b_statevector, create_b_vector_gaussian
from Utils.psi_psi_expect import calculate_psi_psi_from_dict as psi_psi
from Utils.b_psi_expect import calculate_b_psi_from_dict as b_psi
"""
def expectation_with_statevector(ansatz, theta, op):
    #Statevector 시뮬레이션으로 <psi|op|psi> 계산
    psi = Statevector.from_instruction(ansatz.assign_parameters(theta))
    return np.vdot(psi.data, op @ psi.data).real
"""

@lru_cache(maxsize=128)
def _to_dense(op: Union[Operator, SparsePauliOp]) -> np.ndarray:
    return op.to_matrix()

# ────────────────────────────────────────────────────────────
def expectation_with_statevector(
    ansatz: QuantumCircuit,
    theta: Union[Sequence[float], Dict],
    op: Union[Operator, SparsePauliOp, np.ndarray]
) -> float:
    """
    <ψ(θ)| op |ψ(θ)>  를 Statevector 시뮬레이션으로 계산.

    Parameters
    ----------
    ansatz : QuantumCircuit
        매개변수화된 회로.
    theta : Sequence[float] | dict
        ansatz 파라미터 값.
    op : Operator | SparsePauliOp | ndarray
        관측 연산자 (에르미트).

    Returns
    -------
    float
        실수 기대값.
    """
    # 1. 파라미터 바인딩(원본 회로 불변)
    bound_circ = ansatz.assign_parameters(theta, inplace=False)

    # 2. 상태벡터 생성
    psi = Statevector.from_instruction(bound_circ)

    # 3. 기대값 계산 ─ built-in 우선, fallback to dense
    try:
        return float(np.real_if_close(psi.expectation_value(op)))
    except (AttributeError, TypeError):
        op_mat = _to_dense(op) if not isinstance(op, np.ndarray) else op
        return float(np.real_if_close(np.vdot(psi.data, op_mat @ psi.data)))


def run_vqe_for_poisson(m: int, ansatz: 'QuantumCircuit', b_creation_func=create_b_vector_gaussian):
    """VQE 알고리즘을 실행하여 최적의 파라미터를 찾습니다."""
    
    # 1. 해밀토니안 연산자 생성
    A = decompose_A_matrix(m)
    B = decompose_B_matrix(m)
    C_op = dict_to_operator(decompose_C_matrix(m), m)
    #A2_op = B_op - C_op
    
    # 2. 소스 벡터 b 상태 준비
    b_circuit, b_vec, b_normalized = get_b_statevector(num_qubits=m, b_creation_func=b_creation_func)

    
    # 3. 비용 함수 정의
    # <psi | B | psi > - < psi | C | psi > - | <b| A | psi > |^2 
    iteration_count = [0]
    def cost_func(theta: list[float]) -> float:
        iteration_count[0] += 1
        
        term1 = psi_psi(B, ansatz, theta)
        term2 = expectation_with_statevector(ansatz, theta, C_op)
        term3 = b_psi(A, b_circuit, ansatz, theta)

        cost = term1 - term2 - term3
        print(f"Iteration {iteration_count[0]:>4}: Cost = {cost:.8f}", end="\r")
        return cost

    # 4. 최적화 실행
    initial_params = np.random.uniform(0, np.pi, ansatz.num_parameters)
    print()
    print(f"VQE 최적화 시작 (파라미터 수: {ansatz.num_parameters})...")
    
    result = minimize(cost_func, initial_params, method='COBYLA', options={'maxiter': 1000})
    
    print("\n최적화 완료.")
    return result, b_normalized