import numpy as np
from qiskit import QuantumCircuit # type: ignore
from qiskit.primitives import StatevectorEstimator as Estimator # type: ignore
from To_Hermite_Operator import *

def create_real_b_psi_circuit(b_circuit, psi_ansatz, theta_value):
    # 단일 항에 대한 |<b|P|ψ>|² 값을 계산하기 위한 회로 생성
    num_main_qubits = b_circuit.num_qubits
    final_circuit = QuantumCircuit(num_main_qubits + 1)
    main_qubits = list(range(1, num_main_qubits + 1))
    ancilla_qubit = 0
    final_circuit.h(ancilla_qubit)
    
    # .decompose()를 사용하여 회로를 기본 게이트로 분해 (initialize, barrier 등 제거)
    controlled_Ub = b_circuit.decompose().to_gate().control(1, ctrl_state=0)
    final_circuit.append(controlled_Ub, [ancilla_qubit] + main_qubits)

    psi_circuit_with_params = psi_ansatz.assign_parameters(theta_value)
    controlled_Upsi = psi_circuit_with_params.decompose().to_gate().control(1, ctrl_state=1)
    final_circuit.append(controlled_Upsi, [ancilla_qubit] + main_qubits)
    return final_circuit

def create_imag_b_psi_circuit(b_circuit, psi_ansatz, theta_value):
    num_main_qubits = b_circuit.num_qubits
    final_circuit = QuantumCircuit(num_main_qubits + 1)
    main_qubits = list(range(1, num_main_qubits + 1))
    ancilla_qubit = 0
    final_circuit.h(ancilla_qubit)
    final_circuit.s(ancilla_qubit) # 허수부 측정을 위해 S 게이트 추가
    
    controlled_Ub = b_circuit.decompose().to_gate().control(1, ctrl_state=0)
    final_circuit.append(controlled_Ub, [ancilla_qubit] + main_qubits)

    psi_circuit_with_params = psi_ansatz.assign_parameters(theta_value)
    controlled_Upsi = psi_circuit_with_params.decompose().to_gate().control(1, ctrl_state=1)
    final_circuit.append(controlled_Upsi, [ancilla_qubit] + main_qubits)
    return final_circuit



def calculate_b_P_psi_squared(
    p_term: str,
    b_circuit: QuantumCircuit,
    psi_ansatz: QuantumCircuit,
    theta_value: list[float],
    estimator: Estimator
) -> float:
    """
    Hadamard-like 측정을 사용하여 단일 항에 대한 |<b|P|ψ>|² 값을 계산합니다.
    """
    # 1. 측정 가능한 에르미트 관측량 O 생성
    observable = get_hermitian_observable(p_term)

    # 2. 실수부 측정을 위한 상태 준비 및 기댓값 계산
    real_state_circuit = create_real_b_psi_circuit(b_circuit, psi_ansatz, theta_value)
    job_real = estimator.run([(real_state_circuit, [observable], [[]])])
    real_part = job_real.result()[0].data.evs[0]

    # 3. 허수부 측정을 위한 상태 준비 및 기댓값 계산
    imag_state_circuit = create_imag_b_psi_circuit(b_circuit, psi_ansatz, theta_value)
    job_imag = estimator.run([(imag_state_circuit, [observable], [[]])])
    imag_part = job_imag.result()[0].data.evs[0]
    
    # 4. 최종 값 |<b|P|ψ>|² 계산
    magnitude_squared = real_part**2 + imag_part**2

    # print(f"  - Term '{p_term}': Re≈{real_part:.4f}, Im≈{imag_part:.4f}, |<·>|²≈{magnitude_squared:.4f}")
    
    return magnitude_squared



def calculate_b_psi_from_dict(
    p_dict: dict,
    b_circuit: QuantumCircuit,
    psi_ansatz: QuantumCircuit,
    theta_value: list[float]
) -> float:
    """
    연산자 딕셔너리를 받아 각 항의 |<b|P_k|ψ>|² 값을 계산하고,
    계수를 곱하여 총합을 반환합니다.
    """
    total_sum = 0.0
    estimator = Estimator() # Estimator를 한 번만 생성하여 재사용

    #print(f"--- Calculating Sum for Dictionary ---")
    
    for p_term, coeff in p_dict.items():
        # 각 항에 대해 |<b|P|ψ>|² 값을 계산
        term_value_sq = calculate_b_P_psi_squared(
            p_term, b_circuit, psi_ansatz, theta_value, estimator
        )
        # 계산된 값에 계수를 곱하여 총합에 더함
        total_sum += coeff * term_value_sq
        
    #print(f"------------------------------------")
    #print(f"Total Sum = {total_sum:.4f}")
    #print(f'<b|A|ψ> = {total_sum:.4f} ')
    return total_sum