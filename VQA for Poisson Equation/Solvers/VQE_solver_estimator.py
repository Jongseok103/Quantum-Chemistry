# file: VQE_solver_estimator.py

import numpy as np
from scipy.optimize import minimize # type: ignore
from typing import TYPE_CHECKING

# Qiskit 관련 클래스 임포트
from qiskit.primitives import StatevectorEstimator as Estimator # type: ignore
from qiskit.quantum_info import Operator, SparsePauliOp # type: ignore

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit # type: ignore
    
# 다른 모듈에서 필요한 클래스 및 함수 임포트
from Utils.decomposition import decompose_A_matrix, decompose_B_matrix, decompose_C_matrix, dict_to_operator
# get_b_statevector와 기본 b 함수를 임포트합니다.
from Utils.create_b_state import get_b_statevector, create_b_vector_gaussian

def run_vqe_for_poisson(m: int, ansatz: 'QuantumCircuit', b_creation_func=create_b_vector_gaussian):
    """VQE 알고리즘을 Estimator를 사용하여 실행하고 최적의 파라미터를 찾습니다."""
    
    # 1. 해밀토니안 연산자 생성
    A_op = dict_to_operator(decompose_A_matrix(m), m)
    B_op = dict_to_operator(decompose_B_matrix(m), m)
    C_op = dict_to_operator(decompose_C_matrix(m), m)
    
    # A^2 연산자를 SparsePauliOp으로 변환
    A2_op_dense = B_op - C_op
    A2_op = SparsePauliOp.from_operator(A2_op_dense)
    
    # 2. 소스 벡터 b 상태 및 관련 연산자 준비
    #    -> 인자로 받은 b_creation_func를 사용하도록 수정된 부분
    b_vec, b_normalized = get_b_statevector(num_qubits=m, b_creation_func=b_creation_func)
    
    # term2 계산을 위한 연산자 H_b = A^†|b><b|A 생성
    proj_b_op = Operator(b_vec)
    Hb_op_dense = A_op.adjoint().compose(proj_b_op).compose(A_op)
    Hb_op = SparsePauliOp.from_operator(Hb_op_dense)
    
    # Estimator 인스턴스 생성
    estimator = Estimator()

    # 3. 비용 함수 정의
    iteration_count = [0]
    def cost_func(theta: list[float]) -> float:
        iteration_count[0] += 1
        
        # Estimator는 관측 가능 리스트(observables)를 받음
        # 비용 함수: <ψ|A²|ψ> - <ψ|H_b|ψ>
        job = estimator.run([(ansatz, [A2_op, Hb_op], [theta])])
        result = job.result()
        
        exp_vals = result[0].data.evs
        term1_exp_val = exp_vals[0]
        term2_exp_val = exp_vals[1]
        
        cost = term1_exp_val - term2_exp_val
        print(f"Iteration {iteration_count[0]:>4}: Cost = {cost:.8f}", end="\r")
        return cost

    # 4. 최적화 실행
    initial_params = np.random.uniform(0, 2 * np.pi, ansatz.num_parameters)
    print(f"VQE (Estimator) 최적화 시작 (파라미터 수: {ansatz.num_parameters})...")
    
    result = minimize(cost_func, initial_params, method='COBYLA', options={'maxiter': 4000})
    
    print("\n최적화 완료.")
    return result, b_normalized