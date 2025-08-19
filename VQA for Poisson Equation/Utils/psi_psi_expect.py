import numpy as np
from qiskit import QuantumCircuit # type: ignore
from qiskit.primitives import StatevectorEstimator as Estimator # type: ignore
from decomposition import *
from To_Hermite_Operator import *


def measure_non_hermitian_psi_psi(
    op_str: str,
    ansatz: QuantumCircuit,
    theta: list[float],
    estimator: Estimator
) -> complex:
    
    # 연산자 문자열을 에르미트 관측량으로 변환
    num_qubits = len(op_str)
    observable = get_hermitian_observable(op_str)
    
    # --- 실수부 측정을 위한 상태 준비 ---
    real_prep_circuit = QuantumCircuit(num_qubits + 1)
    main_qubits = list(range(1, num_qubits + 1))
    ancilla_qubit = 0
    real_prep_circuit.h(ancilla_qubit)
    real_prep_circuit.compose(ansatz.assign_parameters(theta), qubits=main_qubits, inplace=True)
    
    # --- 허수부 측정을 위한 상태 준비 ---
    imag_prep_circuit = QuantumCircuit(num_qubits + 1)
    imag_prep_circuit.h(ancilla_qubit)
    imag_prep_circuit.s(ancilla_qubit)
    imag_prep_circuit.compose(ansatz.assign_parameters(theta), qubits=main_qubits, inplace=True)
    
    job_real = estimator.run([(real_prep_circuit, [observable], [[]])])
    job_imag = estimator.run([(imag_prep_circuit, [observable], [[]])])
    
    exp_val_real = job_real.result()[0].data.evs[0]
    exp_val_imag = job_imag.result()[0].data.evs[0]
    
    # Re(<P>) = <O>_{real_prep}, Im(<P>) = -<O>_{imag_prep}
    return exp_val_real - 1j * exp_val_imag



def calculate_psi_psi_from_dict(
    p_dict: dict,
    ansatz: QuantumCircuit,
    theta_value: list[float]
) -> complex:
    """
    연산자 딕셔너리 B = Σ c_k P_k에 대해, 기댓값 <ψ|B|ψ>를 계산합니다.

    Args:
        p_dict (dict): {'I+-': coeff1, ...} 형태의 연산자 딕셔너리.
        ansatz (QuantumCircuit): |ψ(θ)⟩ 상태를 만드는 파라미터화된 회로.
        theta_value (list[float]): 안사츠에 할당할 파라미터 값.

    Returns:
        complex: 계산된 복소수 기댓값 <ψ|B|ψ>.
    """
    total_complex_value = 0.0 + 0.0j
    estimator = Estimator() # Estimator를 한 번만 생성하여 재사용
    
    for p_term, coeff in p_dict.items():
        # 각 항에 대해 복소수 기댓값 <ψ|P_k|ψ>를 계산
        complex_exp_val = measure_non_hermitian_psi_psi(
            p_term, ansatz, theta_value, estimator
        )
        #print(f"  - Term '{p_term}': <ψ|P|ψ> ≈ {complex_exp_val:.4f}")
        
        # 계산된 값에 계수를 곱하여 총합에 더함
        total_complex_value += coeff * complex_exp_val
        
    #print(f"------------------------------------")
    #print(f"Total <ψ|O|ψ> ≈ {total_complex_value:.4f}")
    
    # B는 에르미트 연산자이므로, 최종 결과는 실수여야 합니다.
    # 부동 소수점 오차로 인해 작은 허수부가 남을 수 있으므로 실수부만 반환합니다.
    return total_complex_value.real