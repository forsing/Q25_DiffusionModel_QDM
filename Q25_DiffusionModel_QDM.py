#!/usr/bin/env python3
"""
Q25 Diffusion Models — tehnika: Quantum Diffusion Model (QDM)
(čisto kvantno: iterativni reverzni denoiser preko aux-kontrolisanih timestep-tranzicija).

Koncept (kvantni analog klasičnog diffusion modela „noise → denoise"):
  1) Forward referentna serija (deterministička):
        α_t = cos(π·t / (2T)),  t = 0..T.
        f_t = α_t · freq_csv_n + (1 - α_t) · uniform_vec,
        |ψ_t⟩ = amp_from_freq(f_t)  → |ψ_0⟩ = clean CSV state, |ψ_T⟩ = uniformni noise.
  2) Start reverznog procesa: state registar pripremljen u |ψ_T⟩ (maximum noise).
  3) Reverzni denoise (k = 0..T-§1, tj. t ide od T-1 do 0):
        - Fresh aux_k qubit preko Ry(2·β_k) → cos(β_k)|0⟩ + sin(β_k)|1⟩.
        - Controlled-aux (ctrl=1) tranzicioni unitar na state:
              U_{t+1 → t} = SP(amp_t) · SP†(amp_{t+1}).
          Tačno mapira |ψ_{t+1}⟩ → |ψ_t⟩ kad aux=1, identitet kad aux=0.
  4) Marginalizacija SVIH T aux qubit-a → output = kvantna mešavina nad svim
     denoising path-ovima (aux ∈ {0,1}^T).
  5) bias_39 → TOP-7 = NEXT.

Razlika u odnosu na slične fajlove:
  Q14 (Temperature): JEDAN aux, statična mešavina sharp vs uniform.
  Q16 (QAOA):        Hamiltonian-exp operatori (e^{-iγH_P}, e^{-iβH_M}).
  Q21 (QCoT):        sekvencijalni Ry+CNOT lanac BEZ aux-a.
  Q20/Q13:           paralelna superpozicija template-ova / prozora.
  QDM:               T aux qubit-a, iterativne SP·SP† tranzicije između susednih
                     timestep-state-ova, stochastic-path superpozicija nad T denoise-koraka.

Sve deterministički: seed=39; freq_csv iz CELOG CSV-a (pravilo 10).
Deterministička grid-optimizacija (nq, T) po cos(bias_39, freq_csv).

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39

GRID_NQ = (5, 6)
GRID_T = (2, 4, 8)
BETA_SCHEDULE = np.pi / 4


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


def amp_from_freq(f: np.ndarray, nq: int) -> np.ndarray:
    dim = 2 ** nq
    edges = np.linspace(0, N_MAX, dim + 1, dtype=int)
    amp = np.array(
        [float(f[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0 for i in range(dim)],
        dtype=np.float64,
    )
    amp = np.maximum(amp, 0.0)
    n2 = float(np.linalg.norm(amp))
    if n2 < 1e-18:
        amp = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    else:
        amp = amp / n2
    return amp


# =========================
# Forward referentna serija |ψ_t⟩ za t = 0..T
# =========================
def diffusion_amps(H: np.ndarray, nq: int, T: int) -> List[np.ndarray]:
    f_csv = freq_vector(H)
    s = float(f_csv.sum())
    f_csv_n = f_csv / s if s > 0 else np.ones(N_MAX, dtype=np.float64) / N_MAX
    f_uni = np.ones(N_MAX, dtype=np.float64) / N_MAX
    amps: List[np.ndarray] = []
    for t in range(int(T) + 1):
        alpha_t = float(np.cos(np.pi * t / (2.0 * T)))
        f_t = alpha_t * f_csv_n + (1.0 - alpha_t) * f_uni
        amps.append(amp_from_freq(f_t, nq))
    return amps


# =========================
# QDM kolo: |ψ_T⟩ start + T aux-kontrolisanih SP·SP† tranzicija
# =========================
def build_qdm_state(H: np.ndarray, nq: int, T: int) -> Statevector:
    amps = diffusion_amps(H, nq, T)

    state = QuantumRegister(nq, name="s")
    aux = QuantumRegister(int(T), name="a")
    qc = QuantumCircuit(state, aux)

    qc.append(StatePreparation(amps[int(T)].tolist()), state)

    for k in range(int(T)):
        t_from = int(T) - k
        t_to = int(T) - k - 1

        qc.ry(2.0 * BETA_SCHEDULE, aux[k])

        trans = QuantumCircuit(nq, name=f"U_{t_from}_to_{t_to}")
        trans.append(StatePreparation(amps[t_from].tolist()).inverse(), range(nq))
        trans.append(StatePreparation(amps[t_to].tolist()), range(nq))
        trans_gate = trans.to_gate(label=f"trans_k{k}")
        trans_ctrl = trans_gate.control(num_ctrl_qubits=1, ctrl_state=1)
        qc.append(trans_ctrl, [aux[k]] + list(state))

    return Statevector(qc)


def qdm_state_probs(H: np.ndarray, nq: int, T: int) -> np.ndarray:
    sv = build_qdm_state(H, nq, T)
    p = np.abs(sv.data) ** 2
    dim_s = 2 ** nq
    dim_a = 2 ** int(T)
    mat = p.reshape(dim_a, dim_s)
    p_s = mat.sum(axis=0)
    s_tot = float(p_s.sum())
    return p_s / s_tot if s_tot > 0 else p_s


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


# =========================
# Determ. grid-optimizacija (nq, T)
# =========================
def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    s_tot = float(f_csv.sum())
    f_csv_n = f_csv / s_tot if s_tot > 0 else np.ones(N_MAX) / N_MAX
    best = None
    for nq in GRID_NQ:
        for T in GRID_T:
            try:
                p = qdm_state_probs(H, nq, int(T))
                bi = bias_39(p)
                score = cosine(bi, f_csv_n)
            except Exception:
                continue
            key = (score, nq, int(T))
            if best is None or key > best[0]:
                best = (key, dict(nq=nq, T=int(T), score=float(score)))
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q25 Diffusion Model (QDM — reverzni denoise preko aux-path superpozicije): CSV:", CSV_PATH)
    print("redova:", H.shape[0], "| seed:", SEED, "| β:", round(float(BETA_SCHEDULE), 6))

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    print(
        "BEST hparam:",
        "nq=", best["nq"],
        "| T (koraka):", best["T"],
        "| cos(bias, freq_csv):", round(float(best["score"]), 6),
    )

    nq_best = int(best["nq"])
    T_best = int(best["T"])

    f_csv = freq_vector(H)
    s_tot = float(f_csv.sum())
    f_csv_n = f_csv / s_tot if s_tot > 0 else np.ones(N_MAX) / N_MAX

    amps_ref = diffusion_amps(H, nq_best, T_best)
    print("--- forward referentni timesteps (clean → noise) ---")
    for t in range(T_best + 1):
        p_t = np.abs(amps_ref[t]) ** 2
        pred_t = pick_next_combination(p_t)
        cos_t = cosine(bias_39(p_t), f_csv_n)
        alpha_t = float(np.cos(np.pi * t / (2.0 * T_best)))
        print(f"  t={t:d}  α={alpha_t:.4f}  cos(bias, freq_csv)={cos_t:.6f}  NEXT={pred_t}")

    p = qdm_state_probs(H, nq_best, T_best)
    pred = pick_next_combination(p)
    print("--- glavna predikcija (QDM reverzni denoise) ---")
    print("predikcija NEXT:", pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q25 Diffusion Model (QDM — reverzni denoise preko aux-path superpozicije): CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39 | β: 0.785398
BEST hparam: nq= 5 | T (koraka): 2 | cos(bias, freq_csv): 0.901332
--- forward referentni timesteps (clean → noise) ---
  t=0  α=1.0000  cos(bias, freq_csv)=0.900351  NEXT=(7, 19, 22, 24, 27, 28, 31)
  t=1  α=0.7071  cos(bias, freq_csv)=0.901337  NEXT=(7, 19, 22, 24, 27, 28, 31)
  t=2  α=0.0000  cos(bias, freq_csv)=0.902043  NEXT=(1, 2, 3, 4, 5, 6, 7)
--- glavna predikcija (QDM reverzni denoise) ---
predikcija NEXT: (7, 9, x, y, z, 28, 29)
"""



"""
Q25_DiffusionModel_QDM.py — tehnika: Quantum Diffusion Model (QDM).

Koncept:
Reverzni denoising proces kao kvantna superpozicija denoising-path-ova. Forward
referentna serija klasično pre-izračunata: α_t = cos(π·t/(2T)), f_t linearna
mešavina freq_csv i uniform, |ψ_t⟩ amp-encoding. |ψ_0⟩ = clean, |ψ_T⟩ = noise.
Reverzni proces počinje u |ψ_T⟩ i primenjuje T aux-kontrolisanih tranzicija
U_{t+1 → t} = SP(amp_t)·SP†(amp_{t+1}); svaki aux je Ry(2·β) → 50-50 kvantna odluka
„denoise ili ne" za taj korak.

Kolo (nq + T qubit-a):
  StatePreparation(amp_T) na state  (maximum noise start).
  Za k = 0..T−1:
      Ry(π/2) na aux_k  (cos(π/4)|0⟩ + sin(π/4)|1⟩).
      Controlled-aux_k (ctrl=1): SP(amp_{T−k−1}) · SP†(amp_{T−k}) na state.
Readout:
  Marginala aux-registra → p = Σ_path w_path |ψ_path|² → bias_39 → TOP-7 = NEXT.

Tehnike:
Forward reference series preko linearne interpolacije (deterministički scheduler).
StatePreparation i StatePreparation.inverse() za egzaktne timestep-tranzicije.
Aux-kontrolisani unitar preko sub.to_gate().control(ctrl_state=1).
Superpozicija nad svim mogućim denoising sekvencama kroz T fresh aux qubit-a.
Egzaktni Statevector (bez uzorkovanja).
Deterministička grid-optimizacija (nq, T).

Prednosti:
Direktan kvantni analog diffusion modela: reverse-process + scheduled noise level.
Stochastic-path superpozicija (2^T path-ova) razlikuje QDM od Q21 (sekvencijalni bez aux-a)
i od Q14 (1 aux statičan).
Iterativna SP·SP† tranzicija razlikuje QDM od Q20/Q13 (paralelna superpozicija) i od
Q16 (Hamiltonian-exp operatori).
Ceo CSV učestvuje u definiciji |ψ_0⟩ (pravilo 10).
Čisto kvantno: bez klasičnog treninga, bez softmax-a, bez hibrida.

Nedostaci:
Forward referentna serija je klasično pre-izračunata (linearna mešavina freq vs uniform)
— prave diffusion arhitekture uče noise-schedule, ovde je deterministička heuristika.
Fiksna β = π/4 (50-50 po koraku); druge rampe bi dale drugi bias.
Qubit budžet: nq + T — za nq=6, T=8 to je 14 qubit-a (plafon simulacije).
mod-39 readout meša stanja (dim 2^nq ≠ 39).
"""
