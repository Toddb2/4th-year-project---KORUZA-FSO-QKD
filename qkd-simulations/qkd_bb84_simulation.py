"""
qkd_bb84_simulation.py
======================
BB84 QKD channel feasibility simulation for the IRNAS KORUZA FSO testbed.

Simulates BB84 performance (QBER, sifted key rate, secure key rate) as a
function of link distance for three Scottish atmospheric scenarios derived
from libRadtran MYSTIC Monte Carlo simulations.

Photon loss is modelled as an amplitude damping channel in Qiskit.
Secure key rate is computed using the Shor-Preskill bound.

Reference: Blacklaw, T. (2026). Experimental Characterisation of a
Terrestrial Free-Space Optical Link for Quantum Key Distribution.
Heriot-Watt University B.Sc. Year 4 Project (B20PJ).

Dependencies: qiskit==2.3.0, qiskit-aer, numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error

# ── System parameters (Section 3.3 of dissertation) ──────────────────────────
MU          = 0.1       # mean photon number per pulse
ETA_DET     = 0.15      # detector efficiency
F_REP       = 1e6       # pulse repetition rate (Hz)
D_RATE      = 100       # dark count rate (s^-1)
E_MISALIGN  = 0.01      # residual misalignment QBER
SHOTS       = 8192      # Qiskit circuit shots per distance point

# ── Atmospheric scenarios (libRadtran MYSTIC, 1550 nm) ────────────────────────
SCENARIOS = {
    'Clear Summer': 0.0360,   # k_ext (km^-1)
    'Clear Winter': 0.0398,
    'Mist/Drizzle': 0.2368,
}
COLOURS = {
    'Clear Summer': '#2196F3',
    'Clear Winter': '#4CAF50',
    'Mist/Drizzle': '#F44336',
}

# ── Distance sweep ────────────────────────────────────────────────────────────
distances_m  = np.linspace(50, 5000, 200)
distances_km = distances_m / 1000


def eta_atm(k_ext, d_km):
    """Beer-Lambert atmospheric transmittance."""
    return np.exp(-k_ext * d_km)


def qber_theoretical(eta, mu=MU, eta_det=ETA_DET, f_rep=F_REP,
                     d_rate=D_RATE, e_mis=E_MISALIGN):
    """Total QBER: misalignment + dark counts (Eq. 8-9 in dissertation)."""
    e_dark = (d_rate / f_rep) / (2 * mu * eta * eta_det + 1e-12)
    return e_mis + e_dark


def binary_entropy(e):
    """Binary Shannon entropy."""
    e = np.clip(e, 1e-10, 1 - 1e-10)
    return -e * np.log2(e) - (1 - e) * np.log2(1 - e)


def sifted_key_rate(eta, mu=MU, eta_det=ETA_DET, f_rep=F_REP):
    """Sifted key rate (bps), factor 1/2 for basis sifting."""
    return 0.5 * f_rep * mu * eta * eta_det


def secure_key_rate(eta, e):
    """Shor-Preskill secure key rate (Eq. 7 in dissertation)."""
    return sifted_key_rate(eta) * np.maximum(0, 1 - 2 * binary_entropy(e))


def qiskit_qber(eta, shots=SHOTS):
    """
    Amplitude damping circuit: |0> -> H -> damping -> H -> measure.
    Returns fraction of |1> outcomes as QBER estimate.
    """
    gamma = float(np.clip(1 - eta * ETA_DET, 0, 1))
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(amplitude_damping_error(gamma), ['id'])

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.id(0)
    qc.h(0)
    qc.measure(0, 0)

    counts = AerSimulator(noise_model=noise_model).run(qc, shots=shots).result().get_counts()
    return counts.get('1', 0) / shots


def run_simulation():
    results = {}
    circuit_distances = np.array([100, 250, 500, 1000, 2000])

    for name, k_ext in SCENARIOS.items():
        eta_arr  = eta_atm(k_ext, distances_km)
        qber_arr = qber_theoretical(eta_arr)

        c_eta  = eta_atm(k_ext, circuit_distances / 1000)
        c_qber = np.array([qiskit_qber(e) for e in c_eta])

        results[name] = {
            'eta':          eta_arr,
            'qber':         qber_arr,
            'sift_rate':    sifted_key_rate(eta_arr),
            'skr':          secure_key_rate(eta_arr, qber_arr),
            'circuit_dist': circuit_distances,
            'circuit_eta':  c_eta,
            'circuit_qber': c_qber,
        }

        idx = np.argmin(np.abs(distances_m - 500))
        print(f"{name}: eta={eta_arr[idx]:.4f}, QBER={qber_arr[idx]*100:.2f}%, "
              f"Sifted={sifted_key_rate(eta_arr[idx]):.0f} bps, "
              f"SKR={secure_key_rate(eta_arr[idx], qber_arr[idx]):.0f} bps")

    return results


def plot_results(results):
    # ── Figure 1: 4-panel BB84 performance ───────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    titles = ['(a) Atmospheric transmittance', '(b) Quantum bit error rate',
              '(c) Sifted key rate',           '(d) Secure key rate']
    ylabels = ['Transmittance', 'QBER (%)', 'Sifted rate (bps)', 'Secure rate (bps)']
    keys    = ['eta', 'qber', 'sift_rate', 'skr']

    for ax, title, ylabel, key in zip(axes.flatten(), titles, ylabels, keys):
        for name, data in results.items():
            y = data[key] * (100 if key == 'qber' else 1)
            ax.plot(distances_m, y, color=COLOURS[name], lw=2, label=name)
            if key == 'qber':
                ax.scatter(data['circuit_dist'], data['circuit_qber'] * 100,
                           color=COLOURS[name], s=40, zorder=5)
        if key == 'qber':
            ax.axhline(11, color='black', ls=':', lw=1.2, label='11% threshold')
            ax.fill_between(distances_m, 11, 50, alpha=0.08, color='red')
            ax.set_ylim(0, 25)
        ax.axvline(500, color='grey', ls='--', lw=1)
        ax.set(xlabel='Link distance (m)', ylabel=ylabel, title=title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('BB84 QKD Performance — Scottish Atmospheric Scenarios', fontsize=13)
    fig.tight_layout()
    fig.savefig('BB84_QKD_Scottish_Scenarios.png', dpi=200, bbox_inches='tight')
    print("Saved: BB84_QKD_Scottish_Scenarios.png")

    # ── Figure 2: QBER and SKR vs transmittance ───────────────────────────────
    eta_sweep  = np.linspace(0.01, 1.0, 500)
    qber_sweep = qber_theoretical(eta_sweep)
    skr_sweep  = secure_key_rate(eta_sweep, qber_sweep)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(eta_sweep, qber_sweep * 100, 'k', lw=2)
    ax1.axhline(11, color='red', ls='--', lw=1.2, label='11% limit')
    ax1.fill_between(eta_sweep, 11, 50, alpha=0.1, color='red', label='Insecure')
    for name, data in results.items():
        idx = np.argmin(np.abs(distances_m - 500))
        ax1.scatter(data['eta'][idx], data['qber'][idx] * 100,
                    color=COLOURS[name], s=80, zorder=5, label=f"{name} @500m")
    ax1.set(xlabel='Transmittance η', ylabel='QBER (%)',
            title='(a) QBER vs transmittance', ylim=(0, 30))
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(eta_sweep, skr_sweep, 'k', lw=2)
    ax2.axvline(0.033, color='red', ls='--', lw=1.2, label='η ≈ 0.033 (critical)')
    ax2.set(xlabel='Transmittance η', ylabel='Secure key rate (bps)',
            title='(b) Secure key rate vs transmittance')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig('BB84_QBER_vs_Transmittance.png', dpi=200, bbox_inches='tight')
    print("Saved: BB84_QBER_vs_Transmittance.png")
    plt.close('all')


if __name__ == '__main__':
    print("Running BB84 feasibility simulation...")
    results = run_simulation()
    plot_results(results)
    print("Done.")
