from qiskit import QuantumCircuit # type: ignore
from qiskit.circuit import ParameterVector # type: ignore
import numpy as np

def create_qaoa_ansatz(m: int, layers: int = 1) -> QuantumCircuit:
    beta = ParameterVector('β', layers * 2)  # Mixer per qubit
    gamma_zz = ParameterVector('gamma_zz', layers)  # ZZ per pair
    qc = QuantumCircuit(m, name='QAOA_ansatz')
    
    # Initial |+> state
    qc.h(range(m))
    
    for layer in range(layers):
        # Driver HD: ZZ terms (RZ + CNOT)
        for i in range(m):
            j = (i + 1) % m  # Cyclic: i to j
            qc.cx(i, j)
            qc.rz(2 * gamma_zz[layer], j)
            qc.cx(i, j)
        
        #qc.barrier()
        
        # YY on 0-1 (논문 HD)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.cx(0, 1)
        qc.rz(2 * gamma_zz[layer], 1)
        qc.cx(0, 1)
        qc.rx(-np.pi/2, 0)
        qc.rx(-np.pi/2, 1)
        
        #qc.barrier()
        
        # Mixer HM: RX on all
        for i in range(m):
            qc.rx(2 * beta[layer], i)
    
    return qc


