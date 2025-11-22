#!/usr/bin/env python3
"""
Hodgkin–Huxley neuron simulation.
Allows full CLI parameter control and saves figure as hh.png.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def hodgkin_huxley(I_app, time_end, dt,
                   C, g_K, g_Na, g_L,
                   V_K, V_Na, V_L):
    """Run HH simulation and return traces."""
    time = np.arange(0, time_end, dt)

    V = -65.0
    n = 0.0
    m = 0.0
    h = 0.0

    V_trace = []
    I_trace = []

    def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-0.1 * (V + 55)))
    def beta_n(V): return 0.125 * np.exp(-0.0125 * (V + 65))

    def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-0.1 * (V + 40)))
    def beta_m(V): return 4 * np.exp(-0.05 * (V + 65))

    def alpha_h(V): return 0.07 * np.exp(-0.05 * (V + 65))
    def beta_h(V): return 1 / (1 + np.exp(-0.1 * (V + 35)))

    for t in time:
        I = I_app(t)
        I_trace.append(I)

        dn = alpha_n(V) * (1 - n) - beta_n(V) * n
        dm = alpha_m(V) * (1 - m) - beta_m(V) * m
        dh = alpha_h(V) * (1 - h) - beta_h(V) * h

        n += dn * dt
        m += dm * dt
        h += dh * dt

        I_K = g_K * (n**4) * (V - V_K)
        I_Na = g_Na * (m**3) * h * (V - V_Na)
        I_L = g_L * (V - V_L)

        dV = (I - I_K - I_Na - I_L) / C
        V += dV * dt

        V_trace.append(V)

    return time, V_trace, I_trace


def plot_results(time, V_trace, I_trace, out_path):
    """Plot HH simulation results and save."""
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time, V_trace, label="Membrane Potential V (mV)")
    plt.ylabel("V (mV)")
    plt.title("Hodgkin-Huxley Neuron Model")
    plt.legend(loc="upper right")

    plt.subplot(2, 1, 2)
    plt.plot(time, I_trace, label="Input Current I_app", color="orange")
    plt.xlabel("Time (ms)")
    plt.ylabel("I_app")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(
        description="Hodgkin–Huxley neuron model simulation."
    )

    p.add_argument("--time_end", type=float, default=100,
                   help="end of simulation time (ms)")
    p.add_argument("--dt", type=float, default=0.01,
                   help="time step (ms)")
    p.add_argument("--I", type=float, default=10.0,
                   help="constant applied current (uA/cm^2)")

    p.add_argument("--C", type=float, default=1.0,
                   help="membrane capacitance (uF/cm^2)")
    p.add_argument("--g_K", type=float, default=30.0,
                   help="max K+ conductance (mS/cm^2)")
    p.add_argument("--g_Na", type=float, default=120.0,
                   help="max Na+ conductance (mS/cm^2)")
    p.add_argument("--g_L", type=float, default=0.1,
                   help="leak conductance (mS/cm^2)")

    p.add_argument("--V_K", type=float, default=-80.0,
                   help="K+ reversal potential (mV)")
    p.add_argument("--V_Na", type=float, default=50.0,
                   help="Na+ reversal potential (mV)")
    p.add_argument("--V_L", type=float, default=-65.0,
                   help="leak reversal potential (mV)")

    return p.parse_args()


def main():
    print("Starting Hodgkin–Huxley simulation...")
    args = parse_args()
    print("Parameters loaded.")

    I_app = lambda t: args.I

    print("Running dynamics...")
    time, V_trace, I_trace = hodgkin_huxley(
        I_app,
        args.time_end,
        args.dt,
        args.C,
        args.g_K,
        args.g_Na,
        args.g_L,
        args.V_K,
        args.V_Na,
        args.V_L
    )

    out_path = Path(__file__).resolve().parent / "hh.png"
    print(f"Plotting and saving to {out_path} ...")

    plot_results(time, V_trace, I_trace, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
