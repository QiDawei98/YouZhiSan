"""
Mathematical Benchmark: Boundary Handling & Step Size Ratios in High-Dimensional Global Optimization.
Author: Qi Dawei / AI Assistant
Description:
    Evaluates boundary handling operators (clipping, single-step reflection, resampling)
    and perturbation ratios (10%, 20%, 30%) on canonical non-convex multimodal test
    functions (10-D Rastrigin and 10-D Schwefel) using SciPy's Basin-Hopping algorithm.
"""

import os
import sys
import time
import numpy as np
from scipy.optimize import basinhopping


class CustomBoundedStep:
    """
    Uniform boundary-scaled perturbation operator with configurable boundary mode.
    """
    def __init__(self, bounds, step_ratio=0.1, mode='clip'):
        self.bounds = np.array(bounds, dtype=float)
        self.step_ratio = step_ratio
        self.mode = mode
        self.spans = self.bounds[:, 1] - self.bounds[:, 0]
        self.step_sizes = step_ratio * self.spans
        self.boundary_violations = 0
        self.total_steps = 0

    def __call__(self, x):
        self.total_steps += 1
        delta = np.random.uniform(-self.step_sizes, self.step_sizes, size=x.shape)
        x_new = x + delta
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]

        violation = np.any((x_new < low) | (x_new > high))
        if violation:
            self.boundary_violations += 1

        if self.mode == 'clip':
            x_new = np.clip(x_new, low, high)
        elif self.mode == 'reflect':
            # Single-step mirror reflection
            mask_low = x_new < low
            x_new[mask_low] = 2.0 * low[mask_low] - x_new[mask_low]
            mask_high = x_new > high
            x_new[mask_high] = 2.0 * high[mask_high] - x_new[mask_high]
            # Safety clipping against floating-point rounding
            x_new = np.clip(x_new, low, high)
        elif self.mode == 'resample':
            mask = (x_new < low) | (x_new > high)
            if np.any(mask):
                x_new[mask] = np.random.uniform(low[mask], high[mask])
        return x_new


def rastrigin(x):
    """D-dimensional Rastrigin benchmark objective."""
    d = len(x)
    return 10.0 * d + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x))


def schwefel(x):
    """D-dimensional Schwefel benchmark objective."""
    d = len(x)
    return 418.982887 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def run_benchmarks():
    np.random.seed(42)
    n_trials = 30
    n_iter = 50

    benchmarks = [
        {
            'name': 'Rastrigin_10D',
            'func': rastrigin,
            'dim': 10,
            'bounds': [(-5.12, 5.12)] * 10,
        },
        {
            'name': 'Schwefel_10D',
            'func': schwefel,
            'dim': 10,
            'bounds': [(-500.0, 500.0)] * 10,
        }
    ]

    ratios = [0.10, 0.20, 0.30]
    modes = ['clip', 'reflect', 'resample']

    out_lines = []
    header = f"Starting Benchmark: {n_trials} trials per configuration, {n_iter} BH iterations each.\n"
    print(header)
    out_lines.append(header)

    for bench in benchmarks:
        name = bench['name']
        func = bench['func']
        dim = bench['dim']
        bounds = bench['bounds']
        sep = f"================== Benchmark: {name} (Dim={dim}) =================="
        print(sep)
        out_lines.append(sep)

        for ratio in ratios:
            for mode in modes:
                final_funs = []
                violation_rates = []
                durations = []

                for _ in range(n_trials):
                    x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
                    step_taker = CustomBoundedStep(bounds, step_ratio=ratio, mode=mode)

                    t0 = time.time()
                    res = basinhopping(
                        func,
                        x0=x0,
                        niter=n_iter,
                        T=1.0,
                        take_step=step_taker,
                        minimizer_kwargs={
                            'method': 'L-BFGS-B',
                            'bounds': bounds,
                            'options': {'maxiter': 200}
                        }
                    )
                    t1 = time.time()

                    final_funs.append(res.fun)
                    v_rate = step_taker.boundary_violations / max(1, step_taker.total_steps)
                    violation_rates.append(v_rate)
                    durations.append(t1 - t0)

                final_funs = np.array(final_funs)
                v_rates = np.array(violation_rates)
                durations = np.array(durations)

                best_val = np.min(final_funs)
                median_val = np.median(final_funs)
                mean_val = np.mean(final_funs)
                std_val = np.std(final_funs)
                mean_vrate = np.mean(v_rates) * 100.0
                mean_time = np.mean(durations)

                line = (f"Ratio: {int(ratio*100):2d}% | Mode: {mode:8s} | "
                        f"Best: {best_val:10.4f} | Median: {median_val:10.4f} | "
                        f"Mean: {mean_val:10.4f} +/- {std_val:8.4f} | "
                        f"Viol%: {mean_vrate:5.1f}% | Time: {mean_time:.3f}s")
                print(line)
                out_lines.append(line)
        print("\n")
        out_lines.append("\n")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    res_path = os.path.join(out_dir, "results.txt")
    with open(res_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(out_lines))


if __name__ == '__main__':
    run_benchmarks()
