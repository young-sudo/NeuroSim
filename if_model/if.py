#!/usr/bin/env python3
"""
Integrate-and-fire neuron with synaptic conductances.
Runs with default parameters but allows overriding everything via CLI.
Saves figure as if.png in the script directory.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def simulation(V, V_rest, V_th, V_reset, V_inhib,
               w_E, w_I, f_E, f_I,
               S_E, S_I,
               tau_E, tau_I, tau,
               dt, T, with_ap=True):
    """Run neuron simulation and return traces and spike times."""
    time = np.arange(0, T, dt)
    V_trace, S_E_trace, S_I_trace, spike_times = [], [], [], []

    for t in time:
        if (V >= V_th) and with_ap:
            V = V_reset
            spike_times.append(t)

        dS_E = (-S_E / tau_E + (1 - S_E) * f_E) * dt
        dS_I = (-S_I / tau_I + (1 - S_I) * f_I) * dt
        S_E += dS_E
        S_I += dS_I

        dV = (-(V - V_rest) - w_E * S_E * V - w_I * S_I * (V - V_inhib)) / tau * dt
        V += dV

        V_trace.append(V)
        S_E_trace.append(S_E)
        S_I_trace.append(S_I)

    return time, V_trace, S_E_trace, S_I_trace, spike_times


def plot_results(time, V_trace, S_E_trace, S_I_trace, spike_times, V_th, out_path):
    """Plot simulation using original colors and layout."""
    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    plt.plot(time, V_trace, label="Membrane Potential (V)")
    plt.axhline(y=V_th, color='r', linestyle='--', label="Threshold Potential")
    plt.ylabel("V (mV)")
    plt.title("Integrate-and-Fire Neuron Model with Synaptic Conductance")
    plt.xlim(0, time[-1])
    plt.legend(loc='best', bbox_to_anchor=(1, 1), fontsize='xx-small')

    plt.subplot(4, 1, 2)
    plt.plot(time, S_E_trace, color="b")
    plt.xlim(0, time[-1])
    plt.ylabel("Synaptic Activation S_E")

    plt.subplot(4, 1, 3)
    plt.plot(time, S_I_trace, color="r")
    plt.xlim(0, time[-1])
    plt.ylabel("Synaptic Activation S_I")

    plt.subplot(4, 1, 4)
    plt.xlim(0, time[-1])
    plt.eventplot(spike_times, color='black', linelengths=0.8)
    plt.xlabel("Time (ms)")
    plt.ylabel("Spikes")
    plt.title("Spike Times")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(
        description="Integrate-and-fire neuron simulation with synaptic conductances."
    )
    p.add_argument("--T", type=float, default=200.0,
                   help="simulation duration (ms)")
    p.add_argument("--dt", type=float, default=0.1,
                   help="time step (ms)")
    p.add_argument("--V_rest", type=float, default=-65.0,
                   help="resting membrane potential")
    p.add_argument("--V_inhib", type=float, default=-60.0,
                   help="inhibitory reversal potential")
    p.add_argument("--V_th", type=float, default=-45.0,
                   help="spike threshold potential")
    p.add_argument("--V_reset", type=float, default=-90.0,
                   help="reset value after spike")
    p.add_argument("--tau", type=float, default=20.0,
                   help="membrane time constant")
    p.add_argument("--tau_E", type=float, default=5.0,
                   help="excitatory synaptic time constant")
    p.add_argument("--tau_I", type=float, default=10.0,
                   help="inhibitory synaptic time constant")
    p.add_argument("--w_E", type=float, default=0.8,
                   help="excitatory synaptic weight")
    p.add_argument("--w_I", type=float, default=0.1,
                   help="inhibitory synaptic weight")
    p.add_argument("--f_E", type=float, default=0.5,
                   help="excitatory input activation rate")
    p.add_argument("--f_I", type=float, default=0.5,
                   help="inhibitory input activation rate")
    p.add_argument("--no_ap", action="store_true",
                   help="disable action potentials")

    return p.parse_args()


def main():
    print("Starting simulation...")

    args = parse_args()
    print("Parameters loaded.")

    V = args.V_rest
    S_E = 0.0
    S_I = 0.0

    print("Running dynamics...")
    time, V_trace, S_E_trace, S_I_trace, spike_times = simulation(
        V, args.V_rest, args.V_th, args.V_reset, args.V_inhib,
        args.w_E, args.w_I, args.f_E, args.f_I,
        S_E, S_I, args.tau_E, args.tau_I, args.tau,
        args.dt, args.T,
        with_ap=not args.no_ap
    )

    out_path = Path(__file__).resolve().parent / "if.png"
    print(f"Plotting and saving figure to {out_path} ...")
    plot_results(time, V_trace, S_E_trace, S_I_trace, spike_times, args.V_th, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
