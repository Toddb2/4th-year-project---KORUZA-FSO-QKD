"""
scintillation_simulation.py
===========================
Log-normal scintillation fading analysis for BB84 QKD over the IRNAS
KORUZA FSO channel under Scottish atmospheric conditions.

For each (scenario, turbulence regime) pair, N=2000 instantaneous
transmittance samples are drawn from a log-normal distribution. The
QBER and secure key rate are computed for each sample, yielding full
probability distributions. The insecure fraction (P[QBER > 11%]) is
computed as a function of link distance for three turbulence regimes
(SI = 0.1, 0.5, 1.5).

Reference: Blacklaw, T. (2026). Experimental Characterisation of a
Terrestrial Free-Space Optical Link for Quantum Key Distribution.
Heriot-Watt University B.Sc. Year 4 Project (B20PJ).

Dependencies: qiskit==2.3.0, qiskit-aer, numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error

# ── System parameters ─────────────────────────────────────────────────────────
MU         = 0.1
ETA_DET    = 0.15
F_REP      = 1e6
D_RATE     = 100
E_MISALIGN = 0.01
SHOTS      = 2048         # per Qiskit sample point
N_SAMPLES  = 2000         # Monte Carlo samples per (scenario, SI) pair
N_SWEEP    = 500          # samples per distance point in distance sweep
RNG_SEED   = 42
QBER_THRESHOLD = 0.11     # Shor-Preskill security limit

# ── Scenarios and turbulence regimes ──────────────────────────────────────────
SCENARIOS = {
    'Clear Summer': 0.0360,
    'Mist/Drizzle': 0.2368,
}
SI_VALUES  = [0.1, 0.5, 1.5]
SI_LABELS  = {0.1: 'Weak (SI=0.1)', 0.5: 'Moderate (SI=0.5)', 1.5: 'Strong (SI=1.5)'}
SI_COLOURS = {0.1: '#1976D2', 0.5: '#F57C00', 1.5: '#C62828'}

distances_m  = np.linspace(50, 5000, 80)
distances_km = distances_m / 1000


def eta_atm_mean(k_ext, d_km):
    return np.exp(-k_ext * d_km)


def lognormal_samples(eta_mean, si, n=N_SAMPLES, seed=RNG_SEED):
    """
    Draw n instantaneous transmittance samples from log-normal distribution.
    sigma_ln^2 = ln(1 + SI), mu_ln = -sigma_ln^2 / 2  (Andrews & Phillips, 2005).
    Samples clipped to [0, 1].
    """
    rng = np.random.default_rng(seed)
    sigma_ln = np.sqrt(np.log(1 + si))
    mu_ln    = -sigma_ln**2 / 2
    z        = rng.standard_normal(n)
    eta_i    = eta_mean * np.exp(mu_ln + sigma_ln * z)
    return np.clip(eta_i, 0, 1)


def qber(eta_arr, mu=MU, eta_det=ETA_DET, f_rep=F_REP,
         d_rate=D_RATE, e_mis=E_MISALIGN):
    e_dark = (d_rate / f_rep) / (2 * mu * eta_arr * eta_det + 1e-12)
    return e_mis + e_dark


def binary_entropy(e):
    e = np.clip(e, 1e-10, 1 - 1e-10)
    return -e * np.log2(e) - (1 - e) * np.log2(1 - e)


def secure_key_rate(eta_arr, e_arr):
    r_sift = 0.5 * F_REP * MU * eta_arr * ETA_DET
    return r_sift * np.maximum(0, 1 - 2 * binary_entropy(e_arr))


def qiskit_sample_qber(eta, shots=SHOTS):
    """Amplitude damping circuit QBER for a single transmittance sample."""
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


def run_distribution_analysis():
    """
    Compute QBER distributions at 500 m for each (scenario, SI) pair.
    Returns dict of results for plotting.
    """
    results = {}
    print(f"\n{'Scenario':<16} {'SI':>5} {'Mean QBER':>10} {'95th pct':>10} {'Insecure%':>10}")
    print("-" * 55)

    # Qiskit circuit sample points (representative transmittances at 500 m)
    qiskit_etas = np.linspace(0.3, 0.99, 6)
    qiskit_qbers = np.array([qiskit_sample_qber(e) for e in qiskit_etas])

    for scen_name, k_ext in SCENARIOS.items():
        eta_mean_500 = eta_atm_mean(k_ext, 0.5)
        results[scen_name] = {}

        for si in SI_VALUES:
            eta_i   = lognormal_samples(eta_mean_500, si)
            qber_i  = qber(eta_i)
            skr_i   = secure_key_rate(eta_i, qber_i)
            insecure = np.mean(qber_i > QBER_THRESHOLD) * 100

            results[scen_name][si] = {
                'eta_samples':    eta_i,
                'qber_samples':   qber_i,
                'skr_samples':    skr_i,
                'mean_qber':      np.mean(qber_i),
                'p95_qber':       np.percentile(qber_i, 95),
                'insecure_frac':  insecure,
                'qiskit_etas':    qiskit_etas,
                'qiskit_qbers':   qiskit_qbers,
            }

            print(f"{scen_name:<16} {si:>5.1f} {np.mean(qber_i)*100:>9.2f}% "
                  f"{np.percentile(qber_i, 95)*100:>9.2f}% {insecure:>9.2f}%")

    return results


def run_distance_sweep():
    """
    Distance sweep for clear summer: mean QBER, insecure fraction,
    mean SKR vs distance for each turbulence regime.
    """
    k_ext = SCENARIOS['Clear Summer']
    sweep = {si: {'mean_qber': [], 'p95_qber': [], 'insecure': [], 'mean_skr': []}
             for si in SI_VALUES}

    print("\nRunning distance sweep (Clear Summer baseline)...")
    rng = np.random.default_rng(RNG_SEED)

    for d_m, d_km in zip(distances_m, distances_km):
        eta_mean = eta_atm_mean(k_ext, d_km)
        for si in SI_VALUES:
            sigma_ln = np.sqrt(np.log(1 + si))
            mu_ln    = -sigma_ln**2 / 2
            z        = rng.standard_normal(N_SWEEP)
            eta_i    = np.clip(eta_mean * np.exp(mu_ln + sigma_ln * z), 0, 1)
            qber_i   = qber(eta_i)
            skr_i    = secure_key_rate(eta_i, qber_i)

            sweep[si]['mean_qber'].append(np.mean(qber_i))
            sweep[si]['p95_qber'].append(np.percentile(qber_i, 95))
            sweep[si]['insecure'].append(np.mean(qber_i > QBER_THRESHOLD) * 100)
            sweep[si]['mean_skr'].append(np.mean(skr_i))

    for si in SI_VALUES:
        for key in sweep[si]:
            sweep[si][key] = np.array(sweep[si][key])

    return sweep


def plot_distributions(dist_results):
    scenarios = list(dist_results.keys())
    n_rows = len(scenarios)
    n_cols = len(SI_VALUES)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 8))

    for r, scen_name in enumerate(scenarios):
        for c, si in enumerate(SI_VALUES):
            ax  = axes[r, c]
            dat = dist_results[scen_name][si]

            ax.hist(dat['qber_samples'] * 100, bins=60, density=True,
                    color=SI_COLOURS[si], alpha=0.6, edgecolor='none')

            # Log-normal fit
            mu_val  = np.mean(np.log(dat['qber_samples']))
            sig_val = np.std(np.log(dat['qber_samples']))
            x       = np.linspace(dat['qber_samples'].min() * 100,
                                  dat['qber_samples'].max() * 100, 300)
            pdf     = lognorm.pdf(x / 100, s=sig_val, scale=np.exp(mu_val)) / 100
            ax.plot(x, pdf, 'k-', lw=1.5)

            ax.axvline(11, color='red', ls='--', lw=1.2)

            # Qiskit circuit points (only on first row)
            if r == 0:
                ax.scatter(dat['qiskit_qbers'] * 100,
                           [ax.get_ylim()[1] * 0.05] * len(dat['qiskit_qbers']),
                           color='navy', marker='o', s=30, zorder=5,
                           label='Qiskit circuit')

            title = f"{scen_name}\n{SI_LABELS[si]}"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel('QBER (%)', fontsize=8)
            ax.set_ylabel('Probability density', fontsize=8)
            ax.grid(True, alpha=0.25)

            stats_text = (f"Mean: {dat['mean_qber']*100:.2f}%\n"
                          f"95th: {dat['p95_qber']*100:.2f}%\n"
                          f"Insecure: {dat['insecure_frac']:.1f}%")
            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                    fontsize=7, va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    fig.suptitle('QBER Distributions under Log-Normal Scintillation Fading — 500 m',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig('Scintillation_QBER_Distributions.png', dpi=200, bbox_inches='tight')
    print("\nSaved: Scintillation_QBER_Distributions.png")
    plt.close('all')


def plot_distance_sweep(sweep):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    ax1, ax2, ax3 = axes

    for si in SI_VALUES:
        c  = SI_COLOURS[si]
        lb = SI_LABELS[si]
        ax1.plot(distances_m, np.array(sweep[si]['mean_qber']) * 100,
                 color=c, lw=2, label=lb)
        ax1.fill_between(distances_m,
                         np.array(sweep[si]['mean_qber']) * 100,
                         np.array(sweep[si]['p95_qber']) * 100,
                         color=c, alpha=0.15)
        ax2.plot(distances_m, sweep[si]['insecure'], color=c, lw=2, label=lb)
        ax3.plot(distances_m, sweep[si]['mean_skr'], color=c, lw=2, label=lb)

    ax1.axhline(11, color='black', ls=':', lw=1.2)
    for ax in axes:
        ax.axvline(500, color='grey', ls='--', lw=1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Link distance (m)')

    ax1.set(ylabel='QBER (%)',
            title='(a) Mean QBER (band = 95th pct)')
    ax2.set(ylabel='Insecure fraction (%)',
            title='(b) Fraction of time link insecure')
    ax3.set(ylabel='Mean secure rate (bps)',
            title='(c) Mean secure key rate')

    fig.suptitle('FSO QKD vs Distance — Log-Normal Scintillation (Clear Summer)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig('Scintillation_Distance_Sweep.png', dpi=200, bbox_inches='tight')
    print("Saved: Scintillation_Distance_Sweep.png")
    plt.close('all')


if __name__ == '__main__':
    print("Running scintillation fading analysis...")
    dist_results = run_distribution_analysis()
    plot_distributions(dist_results)
    sweep_results = run_distance_sweep()
    plot_distance_sweep(sweep_results)
    print("Done.")
