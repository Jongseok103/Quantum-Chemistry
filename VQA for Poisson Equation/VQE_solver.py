# file: VQE_solver.py

import numpy as np
from qiskit.quantum_info import Statevector # type: ignore
from scipy.optimize import minimize # type: ignore
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit # type: ignore

# 다른 모듈에서 필요한 클래스 및 함수 임포트
from decomposition import decompose_A_matrix, decompose_B_matrix, decompose_C_matrix, dict_to_operator
from create_b_state import get_b_statevector, create_b_vector_gaussian

def run_vqe_for_poisson(m: int, ansatz: 'QuantumCircuit', b_creation_func=create_b_vector_gaussian):
    """VQE 알고리즘을 실행하여 최적의 파라미터를 찾습니다."""
    
    # 1. 해밀토니안 연산자 생성
    A_op = dict_to_operator(decompose_A_matrix(m), m)
    B_op = dict_to_operator(decompose_B_matrix(m), m)
    C_op = dict_to_operator(decompose_C_matrix(m), m)
    A2_op = B_op - C_op
    
    # 2. 소스 벡터 b 상태 준비
    b_vec, b_normalized = get_b_statevector(num_qubits=m, b_creation_func=b_creation_func)

    
    # 3. 비용 함수 정의
    iteration_count = [0]
    def cost_func(theta: list[float]) -> float:
        iteration_count[0] += 1
        
        psi_vec = Statevector.from_instruction(ansatz.assign_parameters(theta))
        
        term1 = psi_vec.expectation_value(A2_op).real
        evolved_psi_vec = psi_vec.evolve(A_op)
        z = b_vec.inner(evolved_psi_vec)
        term2 = np.abs(z)**2
        
        cost = term1 - term2
        print(f"Iteration {iteration_count[0]:>4}: Cost = {cost:.8f}", end="\r")
        return cost

    # 4. 최적화 실행
    initial_params = np.random.uniform(0, 2 * np.pi, ansatz.num_parameters)
    print(f"VQE 최적화 시작 (파라미터 수: {ansatz.num_parameters})...")
    
    result = minimize(cost_func, initial_params, method='COBYLA', options={'maxiter': 4000})
    
    print("\n최적화 완료.")
    return result, b_normalized