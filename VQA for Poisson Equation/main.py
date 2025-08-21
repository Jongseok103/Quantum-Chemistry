# file: main.py

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit # type: ignore
from qiskit.quantum_info import Statevector # type: ignore
import scipy.linalg as la # type: ignore

# 각 모듈에서 필요한 함수들을 임포트합니다.
from Solvers.VQE_solver import run_vqe_for_poisson # type: ignore
from Solvers.VQE_solver_estimator import run_vqe_for_poisson as run_vqe_estimator
from Solvers.VQE_solver3 import run_vqe_for_poisson as run_vqe3


from Utils.Ansatz import create_qaoa_ansatz as create_ansatz
from Utils.decomposition import decompose_A_matrix, dict_to_operator
from Utils.create_b_state import (
    create_b_vector_gaussian,
    create_b_vector_sine,
    create_b_vector_uniform,
    create_b_vector_random,
    create_b_vector_linear
)

def get_user_inputs():
    """사용자로부터 시뮬레이션 파라미터를 입력받습니다."""
    m_qubits = int(input("Enter the number of qubits (e.g., 3): "))
    ansatz_layers = int(input("Enter the number of ansatz layers (e.g., 4): "))
    
    # b 함수 선택
    b_functions = {
        '1': ('Gaussian', create_b_vector_gaussian), '2': ('Sine', create_b_vector_sine),
        '3': ('Uniform', create_b_vector_uniform), '4': ('Random', create_b_vector_random),
        '5': ('Linear', create_b_vector_linear),
    }
    print("\nSelect the function to generate vector b:")
    for key, (name, _) in b_functions.items(): print(f"  {key}: {name}")
    choice = input(f"Enter your choice (1-{len(b_functions)}): ")
    b_func_name, b_func = b_functions.get(choice, ('Gaussian', create_b_vector_gaussian))
    
    # Solver 선택
    print("\nSelect the VQE solver to use:")
    print("  1: Statevector Solver (fast, exact simulation)")
    print("  2: Estimator Primitive (slower, simulates backend execution)")
    print("  3: paper-style cost: <psi|B|psi>-<psi|C|psi>-|<b|A|psi>|^2")
    while True:
        solver_choice = input("Enter your choice (1 or 2 or 3): ")
        if solver_choice in ['1', '2', '3' ]:
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
            
    return m_qubits, ansatz_layers, b_func, b_func_name, solver_choice

def get_classical_solution(m: int, b_creation_func) -> np.ndarray:
    """고전적인 방법으로 푸아송 방정식의 해를 계산하고 정규화합니다."""
    n = 2**m
    A_op = dict_to_operator(decompose_A_matrix(m), m)
    A_matrix = A_op.to_matrix()
    
    # 원본 b 벡터 생성
    x_grid = np.linspace(0, 1, n + 2)[1:-1]
    if "gaussian" in b_creation_func.__name__:
        sigma, delta_center = 0.2, 0.5
        b_vec_unnormalized = np.exp(-(x_grid - delta_center)**2 / (2 * sigma**2))
    elif "sine" in b_creation_func.__name__:
        b_vec_unnormalized = np.sin(2 * np.pi * x_grid)
    elif "uniform" in b_creation_func.__name__:
        b_vec_unnormalized = np.ones(n)
    elif "random" in b_creation_func.__name__:
        b_vec_unnormalized = b_creation_func(n) * np.sqrt(n) # non-deterministic
    elif "linear" in b_creation_func.__name__:
        b_vec_unnormalized = x_grid
    else:
        b_vec_unnormalized = b_creation_func(n) * np.sqrt(n)

    x_exact = la.solve(A_matrix, b_vec_unnormalized)
    norm = np.linalg.norm(x_exact)
    return x_exact / norm if norm > 0 else x_exact

def calculate_fidelity(state_vqe: Statevector, state_exact: np.ndarray) -> float:
    """VQE 상태와 정확한 해 상태 간의 피델리티를 계산합니다."""
    psi_exact = Statevector(state_exact)
    return np.abs(psi_exact.inner(state_vqe))**2

def infer_solution_statevector(ansatz: QuantumCircuit, optimal_params: np.ndarray) -> Statevector:
    """VQE 결과로부터 최종 해의 상태벡터를 추론합니다."""
    solution_circuit = ansatz.assign_parameters(optimal_params)
    return Statevector.from_instruction(solution_circuit)

def plot_x_exact_vs_x_vqe(m: int, x_exact_vector: np.ndarray, x_vqe_vector: np.ndarray, fidelity: float, title_suffix: str):
    """실제 해와 VQE 해를 비교하여 시각화합니다."""
    n = 2**m
    grid_points = np.linspace(0, 1, n + 2)[1:-1]
    
    plt.figure(figsize=(10, 6))
    if np.dot(x_vqe_vector.real, x_exact_vector.real) < 0: x_vqe_vector = -x_vqe_vector
        
    plt.plot(grid_points, x_exact_vector.real, 'o--', lw=2, label='Exact Solution (Classical)')
    plt.plot(grid_points, x_vqe_vector.real, 's-', lw=2, alpha=0.7, label='VQE Solution (Quantum)')
    plt.title(f'Comparison of Exact and VQE Solutions\n({title_suffix})', fontsize=14)
    plt.xlabel('Position'); plt.ylabel('Normalized Amplitude')
    plt.legend()
    plt.suptitle(f'Fidelity = {fidelity:.6f}', fontsize=16, y=1.02)
    plt.grid(True, linestyle='--')
    plt.savefig("exact_vs_vqe_comparison.png")
    plt.show()

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    # 1. 파라미터 입력 (솔버 선택 포함)
    m_qubits, ansatz_layers, b_func, b_func_name, solver_choice = get_user_inputs()
    
    # 2. 안사츠 생성
    ansatz_circuit = create_ansatz(m=m_qubits, layers=ansatz_layers)
    
    
    # 3. 사용자 선택에 따라 VQE 실행
    
    solver_name = ""
    if solver_choice == '1':
        solver_name = "Statevector Solver"
        vqe_result, b_vec_normalized = run_vqe_for_poisson(
            m=m_qubits, ansatz=ansatz_circuit, b_creation_func=b_func
        )

    elif solver_choice == '2':
        solver_name = "Estimator Primitive Solver"
        vqe_result, b_vec_normalized = run_vqe_estimator(
            m=m_qubits, ansatz=ansatz_circuit, b_creation_func=b_func
        )
    else: 
        # solver_choice == '3'
        solver_name = "Paper-style Cost Solver"
        vqe_result, b_vec_normalized = run_vqe3(
            m=m_qubits, ansatz=ansatz_circuit, b_creation_func=b_func
        )

    # 4. VQE 해 추론
    optimal_parameters = vqe_result.x
    solution_state_vqe = infer_solution_statevector(ansatz_circuit, optimal_parameters)
    solution_x_vqe = solution_state_vqe.data

    
    # 5. 고전적 해 및 피델리티 계산
    exact_solution_vector = get_classical_solution(m_qubits, b_func)
    fidelity = calculate_fidelity(Statevector(solution_x_vqe), exact_solution_vector)

    if np.dot(solution_x_vqe, exact_solution_vector.real) < 0:
        solution_x_vqe *= -1
    solution_x_vqe /= np.linalg.norm(solution_x_vqe)

    
    # 6. 결과 출력
    title = f"{m_qubits}-qubit, {ansatz_layers}-layer, b-func: {b_func_name}, Solver: {solver_name}"
    print("\n" + "="*80 + f"\nVQE Final Results ({title})\n" + "="*80)
    print(f"▷ Minimum Cost (Functional Value): {vqe_result.fun:.8f}")
    print(f"▷ Fidelity with classical solution: {fidelity:.8f}  (Ideal = 1.0)")
    print("="*80)
    
    # 7. 그래프 시각화
    plot_x_exact_vs_x_vqe(
        m=m_qubits, x_exact_vector=exact_solution_vector, 
        x_vqe_vector=solution_x_vqe, fidelity=fidelity, title_suffix=title
    )