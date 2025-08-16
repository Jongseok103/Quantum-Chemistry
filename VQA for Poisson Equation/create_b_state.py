# file: create_b_state.py

import numpy as np
from qiskit import QuantumCircuit # type: ignore
from qiskit.quantum_info import Statevector # type: ignore

def create_b_vector_gaussian(num_points: int) -> np.ndarray:
    """가우시안 형태의 소스 벡터 b를 생성하고 정규화합니다."""
    x = np.linspace(0, 1, num_points + 2)[1:-1]
    sigma, delta_center = 0.2, 0.5
    b = np.exp(-(x - delta_center)**2 / (2 * sigma**2))
    norm = np.linalg.norm(b)
    return b / norm if norm > 0 else b


def create_b_vector_sine(num_points: int) -> np.ndarray:
    """사인 함수 형태의 소스 벡터 b를 생성하고 정규화합니다."""
    x = np.linspace(0, 1, num_points + 2)[1:-1]
    b = np.sin(2 * np.pi * x)
    norm = np.linalg.norm(b)
    return b / norm if norm > 0 else b


def create_b_vector_uniform(num_points: int) -> np.ndarray:
    """균등 분포 형태의 소스 벡터 b를 생성하고 정규화합니다."""
    b = np.ones(num_points)
    norm = np.linalg.norm(b)
    return b / norm if norm > 0 else b

def create_b_vector_random(num_points: int) -> np.ndarray:
    """랜덤 형태의 소스 벡터 b를 생성하고 정규화합니다."""
    b = np.random.rand(num_points)
    norm = np.linalg.norm(b)
    return b / norm if norm > 0 else b

def create_b_vector_linear(num_points: int) -> np.ndarray:
    """선형 형태의 소스 벡터 b를 생성하고 정규화합니다."""
    x = np.linspace(0, 1, num_points + 2)[1:-1]
    b = x
    norm = np.linalg.norm(b)
    return b / norm if norm > 0 else b

def get_b_statevector(num_qubits: int, b_creation_func=create_b_vector_gaussian) -> tuple[Statevector, np.ndarray]:
    """
    지정된 함수를 사용해 b 벡터와 그에 해당하는 양자 상태 벡터를 생성합니다.
    
    Args:
        num_qubits (int): 큐비트 수.
        b_creation_func (function): b 벡터를 생성하는 함수.
    
    Returns:
        tuple[Statevector, np.ndarray]: b에 해당하는 상태 벡터와 정규화된 b 벡터.
    """
    num_points = 2**num_qubits
    b_normalized = b_creation_func(num_points)
    
    b_circuit = QuantumCircuit(num_qubits)
    b_circuit.initialize(b_normalized, range(num_qubits))
    b_vec = Statevector.from_instruction(b_circuit)
    
    return b_vec, b_normalized