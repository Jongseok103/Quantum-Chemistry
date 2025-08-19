# file: VQE_solver.py

import numpy as np
from typing import TYPE_CHECKING, Literal, Callable, Tuple, Optional

from qiskit import QuantumCircuit # type: ignore
from qiskit.quantum_info import Statevector, Operator # type: ignore

from scipy.optimize import minimize  # type: ignore

from decomposition import (
    decompose_A_matrix,
    decompose_B_matrix,
    decompose_C_matrix,
    dict_to_operator,
)
from create_b_state import get_b_statevector, create_b_vector_gaussian

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit as QC # type: ignore


def _hadamard_test_overlap_real(
    ansatz: "QC",
    theta: np.ndarray,
    A_op: Operator,
    b_state: Statevector,
) -> float:
    """
    Re(<b| A |psi(theta)>)를 Hadamard test 스타일로 추정.
    회로:
      - ancilla |0> -> H -> |+>
      - system: prepare |psi(theta)>
      - controlled-A (control: ancilla, target: system)
      - (b-state와의 오버랩을 위한 역유니터리)  : U_b^\dagger 를 system에 적용 (state |b> -> |0...0>)
      - ancilla 측정의 Z 기대값 -> Re(overlap)

    구현상 단순화를 위해:
      - b_state 를 prepare하는 유니터리 U_b 를 회로로 직접 구성하는 대신,
        U_b^\dagger 대신 b_state.dag().data 를 맵핑하기 위한 unitary를 사용하지 않고,
        아래와 같이 trick을 사용:
          Re(<b|A|psi>) = Re( (b^\dagger) (A |psi>) )
        즉, statevector 방식으로 시스템 레지스터에만 A|psi>를 만들어 ancilla Z측정을 회로 없이
        재현하려면 full statevector 시뮬이 필요하므로, 본 함수는 회로 기반으로 다음을 수행:
          - ancilla와 시스템을 합친 전체 회로에서 마지막에 b-basis 측정 대신
            b를 computational basis로 매핑하는 unitary를 적용.
    주의: qiskit에서 임의 Statevector를 정확히 prepare하는 circuit synthesis는 heavy.
    여기서는 Operator로 unitary 삽입(Operator(b_state.to_operator().data)) 접근 사용.

    반환: 실수부 추정값 (exact simulator에서 expectation으로 계산)
    """
    n = ansatz.num_qubits
    total = QuantumCircuit(1 + n)

    # 1) ancilla H -> |+>
    total.h(0)

    # 2) system prepare |psi(theta)>
    psi_circ = ansatz.assign_parameters(theta)
    total.compose(psi_circ, qubits=range(1, 1 + n), inplace=True)

    # 3) controlled-A on system (control ancilla)
    #    A_op는 n-qubit Operator로 가정
    A_gate = A_op  # already Operator
    # controlled 적용을 위해, 먼저 A를 유니터리 게이트로 삽입
    # qiskit는 Operator의 controlled()를 지원
    cA_gate = A_gate.control()
    total.append(cA_gate, qargs=[0] + list(range(1, 1 + n)))

    # 4) U_b^\dagger 적용: b_state를 |0...0>으로 맵핑하는 유니터리의 dagger
    #    statevector를 unitary로 만드는 쉬운 방법: Operator(|b><b| + ... )는 projector라 unitary 아님.
    #    대신 initialize는 unitary 아님. 여기서는 정확 시뮬(Statevector) 사용 경로에서만 쓰는 것이 안전.
    #    따라서 여기의 회로 버전은 실제 하드웨어 대상이 아니라 정확 시뮬에서만 동작하도록
    #    Statevector evolve/내적으로 계산하는 별도 경로를 제공.
    raise NotImplementedError("회로 기반 U_b 구현은 heavy합니다. 아래 statevector 경로를 사용하세요.")


def _overlap_real_imag_statevector(
    ansatz: "QC",
    theta: np.ndarray,
    A_op: Operator,
    b_state: Statevector,
) -> Tuple[float, float]:
    """
    정확 시뮬(상태벡터)로 Re, Im(<b|A|psi(theta)>) 계산.
    """
    psi_vec = Statevector.from_instruction(ansatz.assign_parameters(theta))
    evolved = psi_vec.evolve(A_op)
    z = b_state.inner(evolved)  # <b| (A|psi>)
    return float(np.real(z)), float(np.imag(z))


def _term2_value_from_overlap(re_val: float, im_val: float) -> float:
    return re_val * re_val + im_val * im_val


def run_vqe_for_poisson(
    m: int,
    ansatz: "QC",
    b_creation_func: Callable = create_b_vector_gaussian,
    term2_mode: Literal["statevector", "ancilla"] = "statevector",
    optimizer: Literal["COBYLA", "Nelder-Mead", "SPSA"] = "COBYLA",
    maxiter: int = 4000,
    random_seed: Optional[int] = None,
):
    """
    VQE 알고리즘 실행.
    - term2_mode
      "statevector": 임시 경로(기본). Statevector로 |<b|A|psi>|^2 정확 계산.
      "ancilla": 권장 경로. ancilla+controlled-A+Bell 아이디어로 회로 기반 (현재 NotImplementedError).
                 실제 백엔드용으로 확장하려면 U_b 준비/역연산 회로 합성이 필요.

    반환: (scipy.optimize 결과, 정규화된 b 벡터)
    """

    # 1. 연산자 구성
    A_op = dict_to_operator(decompose_A_matrix(m), m)  # n-qubit Operator
    B_op = dict_to_operator(decompose_B_matrix(m), m)
    C_op = dict_to_operator(decompose_C_matrix(m), m)
    A2_op = B_op - C_op

    # 2. b 상태와 벡터
    b_state, b_normalized = get_b_statevector(num_qubits=m, b_creation_func=b_creation_func)

    rng = np.random.default_rng(random_seed)
    initial_params = rng.uniform(0, 2 * np.pi, ansatz.num_parameters)

    iteration_count = [0]

    def cost_func(theta: np.ndarray) -> float:
        iteration_count += 1
        # term1 = <psi|A^2|psi>
        psi_vec = Statevector.from_instruction(ansatz.assign_parameters(theta))
        term1 = float(np.real(psi_vec.expectation_value(A2_op)))

        # term2 = |<b|A|psi>|^2
        if term2_mode == "statevector":
            re_val, im_val = _overlap_real_imag_statevector(ansatz, theta, A_op, b_state)
            term2 = _term2_value_from_overlap(re_val, im_val)
        elif term2_mode == "ancilla":
            # 회로 기반 경로(권장): U_b 합성 필요. 현재는 NotImplementedError 처리.
            raise NotImplementedError(
                "ancilla 경로는 U_b 합성(임의 상태 준비/역연산) 회로 구현이 필요합니다. "
                "실험적으로는 statevector 모드를 사용하거나, U_b 합성기를 추가하세요."
            )
        else:
            raise ValueError("Unknown term2_mode")

        cost = term1 - term2
        print(f"Iteration {iteration_count[0]:>4}: Cost = {cost:.8f}", end="\r")
        return cost

    print(f"VQE 최적화 시작 (파라미터 수: {ansatz.num_parameters}, term2_mode={term2_mode})...")
    result = minimize(cost_func, initial_params, method=optimizer, options={"maxiter": maxiter})
    print("\n최적화 완료.")

    return result, b_normalized
