# file: analysis_plotter.py

import numpy as np
import matplotlib.pyplot as plt
import time

# (이전과 동일한 임포트 및 함수 정의)
from VQE_solver import run_vqe_for_poisson
from VQE_solver_estimator import run_vqe_for_poisson as run_vqe_estimator
from Ansatz import create_qaoa_ansatz as create_ansatz
from decomposition import decompose_A_matrix, dict_to_operator
from create_b_state import create_b_vector_gaussian
from main import get_classical_solution, calculate_fidelity, infer_solution_statevector

def run_fidelity_experiment(qubit_list, layer_range, b_func):
    results = {}
    print("="*60)
    print("Starting VQE Fidelity vs. Layers Analysis")
    print(f"Qubit Counts to Test: {qubit_list}")
    print(f"Layer Counts to Test: {list(layer_range)}")
    print("="*60)
    total_start_time = time.time()
    for m_qubits in qubit_list:
        print(f"\n[INFO] Running simulations for {m_qubits} qubits...")
        fidelities_for_qubit = []
        for layers in layer_range:
            layer_start_time = time.time()
            ansatz_circuit = create_ansatz(m=m_qubits, layers=layers)

            
            vqe_result, _ = run_vqe_estimator(
                m=m_qubits, 
                ansatz=ansatz_circuit, 
                b_creation_func=b_func
            )
            optimal_params = vqe_result.x
            solution_state = infer_solution_statevector(ansatz_circuit, optimal_params)
            exact_solution = get_classical_solution(m_qubits, b_func)
            fidelity = calculate_fidelity(solution_state, exact_solution)
            fidelities_for_qubit.append(fidelity)
            layer_end_time = time.time()
            print(f"  -> {m_qubits} Qubits, {layers} Layers: Fidelity = {fidelity:.6f} "
                  f"(took {layer_end_time - layer_start_time:.2f} s)")
        results[m_qubits] = fidelities_for_qubit
    total_end_time = time.time()
    print(f"\n[INFO] All simulations finished in {total_end_time - total_start_time:.2f} seconds.")
    return results

def plot_results(results, layer_range):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))
    for m_qubits, fidelities in results.items():
        plt.plot(
            layer_range, 
            fidelities, 
            marker='o', 
            linestyle='--', 
            label=f'{m_qubits} Qubits'
        )
    plt.title('VQE Performance: Fidelity vs. Ansatz Layers', fontsize=18)
    plt.xlabel('Number of Ansatz Layers', fontsize=14)
    plt.ylabel('Fidelity with Exact Solution', fontsize=14)
    plt.xticks(layer_range)
    plt.ylim(0.9, 1.05)
    plt.legend(title='Number of Qubits', fontsize=12)
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.savefig('fidelity_vs_layers_analysis.png')
    print("\n[SUCCESS] Analysis plot saved to 'fidelity_vs_layers_analysis.png'")
    plt.show()

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    # === 분석 파라미터 설정 ===

    # 테스트할 큐빗 수 목록 (2개부터 8개까지)
    # 경고: 6개 이상은 매우 오랜 시간이 소요됩니다. [2, 3, 4, 5]로 시작하는 것을 권장합니다.
    QUBIT_LIST_TO_TEST = list(range(2, 5))
    
    # 테스트할 레이어 수 범위 (1부터 8까지)
    LAYER_RANGE_TO_TEST = range(1, 9)
    
    # 분석에 사용할 b 함수 (일관된 비교를 위해 하나로 고정)
    B_FUNCTION_FOR_TEST = create_b_vector_gaussian
    
    # 1. 실험 실행
    fidelity_results = run_fidelity_experiment(
        qubit_list=QUBIT_LIST_TO_TEST,
        layer_range=LAYER_RANGE_TO_TEST,
        b_func=B_FUNCTION_FOR_TEST
    )
    
    # 2. 결과 시각화
    plot_results(fidelity_results, LAYER_RANGE_TO_TEST)