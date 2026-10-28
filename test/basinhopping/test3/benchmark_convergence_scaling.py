"""
Basin-Hopping Iteration Scaling & Deep Convergence Benchmark for YouZhiSan.
Author: Qi Dawei / AI Assistant
Description:
    Evaluates convergence trajectories, barrier crossing capability, and residual scaling
    across iteration budgets (niter = 20, 40, 60) on representative rare-earth
    crystal field Hamiltonians.
"""

import os
import sys
import re
import copy
import time
import json
import random
from multiprocessing import Pool, cpu_count

# Setup environment paths relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
test_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

site_pkg = r"C:\ProgramData\anaconda3\envs\PyCrystalField\Lib\site-packages"
if site_pkg not in sys.path:
    sys.path.insert(0, site_pkg)

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

if test_root not in sys.path:
    sys.path.insert(0, test_root)

import numpy as np
from scipy.optimize import basinhopping
import YZS_lib
from lbfgsb_benchmark.benchmark_lbfgsb import parse_real_case


# ==============================================================================
# 1. Step Operator Implementation
# ==============================================================================
class BenchmarkBoundedStep:
    """
    Uniform boundary-scaled perturbation operator with mirror reflection or clipping.
    """
    def __init__(self, bounds, step_ratio=0.20, mode='reflect', seed=None):
        self.bounds = np.array(bounds, dtype=float)
        self.step_ratio = float(step_ratio)
        self.step_sizes = self.step_ratio * (self.bounds[:, 1] - self.bounds[:, 0])
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.boundary_violations = 0
        self.total_steps = 0

    def __call__(self, x):
        self.total_steps += 1
        delta = self.rng.uniform(-self.step_sizes, self.step_sizes, size=x.shape)
        x_new = x + delta
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]

        if np.any((x_new < low) | (x_new > high)):
            self.boundary_violations += 1

        if self.mode == 'reflect':
            # Vectorized single-step mirror reflection
            mask_low = x_new < low
            x_new[mask_low] = 2.0 * low[mask_low] - x_new[mask_low]
            mask_high = x_new > high
            x_new[mask_high] = 2.0 * high[mask_high] - x_new[mask_high]
            np.clip(x_new, low, high, out=x_new)
        else:
            np.clip(x_new, low, high, out=x_new)

        return x_new


# ==============================================================================
# 2. Key Configurations to Test
# ==============================================================================
CONFIGURATIONS = [
    {"name": "10%_clip (Baseline)",  "ratio": 0.10, "mode": "clip"},
    {"name": "15%_reflect",           "ratio": 0.15, "mode": "reflect"},
    {"name": "20%_clip",              "ratio": 0.20, "mode": "clip"},
    {"name": "20%_reflect (Target)",  "ratio": 0.20, "mode": "reflect"},
    {"name": "25%_reflect",           "ratio": 0.25, "mode": "reflect"},
]

CHECKPOINTS = [20, 40, 60]
TOTAL_NITER = 60


# ==============================================================================
# 3. Problem Instances Loading
# ==============================================================================
REAL_TEST_FILES = [
    r"(Gruber J B et al, 1960)\#Gruber J B, Conway J G, 1960. Crys.txt",
    r"(Wong E Y et al, 1961)\(Wong E Y et al, 1961).txt",
    r"(Crosswhite H M et al, 1961)\(Crosswhite H M et al, 1961).txt",
    r"(Erath E H, 1961)\(Erath E H, 1961).txt",
    r"(Judd B R et al, 1957)\(Judd B R et al, 1957).txt",
    r"(Koningstein J A et al, 1964)\(Koningstein J A et al, 1964).txt",
    r"(Capobianco J A et al, 1997)\(Capobianco J A et al, 1997).txt",
    r"(Popova M N et al, 2007)\(Popova M N et al, 2007).txt",
    r"(Chen X Y et al, 2005)\(Chen X Y et al, 2005).txt",
]


def load_real_cases():
    cases = []
    for rel in REAL_TEST_FILES:
        full_p = os.path.join(test_root, rel)
        if not os.path.exists(full_p):
            continue
        c = parse_real_case(full_p)
        if c:
            cases.append(c)
    return cases


def generate_synthetic_cases(seed=2026):
    random.seed(seed)
    np.random.seed(seed)

    target_symmetries = ['C6V', 'D4', 'D2D', 'D3', 'C2V']
    available_ions = sorted(list(set(k[0] for k in YZS_lib.Thet.keys())))
    synthetic_cases = []

    for sym in target_symmetries:
        Ar_dict = YZS_lib.allowed_cfp(sym)
        created = 0
        attempts = 0
        while created < 2 and attempts < 100:
            attempts += 1
            ion = random.choice(available_ions)
            valid_terms = [k[1] for k in YZS_lib.Thet.keys() if k[0] == ion]
            if not valid_terms:
                continue
            terms = random.sample(valid_terms, min(2, len(valid_terms)))

            p_star = []
            bounds = []
            for trm in Ar_dict.keys():
                if trm.startswith('AR2'):
                    val = float(np.random.uniform(-120.0, 120.0))
                    bnd = (-200.0, 200.0)
                elif trm.startswith('AR4'):
                    val = float(np.random.uniform(-80.0, 80.0))
                    bnd = (-150.0, 150.0)
                else:
                    val = float(np.random.uniform(-40.0, 40.0))
                    bnd = (-100.0, 100.0)
                p_star.append(val)
                bounds.append(bnd)
            p_star = np.array(p_star, dtype=float)

            try:
                th_nrg = YZS_lib.THNRG(ion, Ar_dict.copy(), p_star, terms, False)
            except Exception:
                continue

            valid = True
            EXNRG = {}
            for trm in terms:
                m = re.search(r'\d+[A-Z](\d+\/\d+|\d+)', trm)
                J_str = m.group(1)
                J = (int(J_str.split('/')[0]) / int(J_str.split('/')[1])) if '/' in J_str else int(J_str)
                expected_count = YZS_lib.get_stark_levels_count(sym, J)
                if len(th_nrg[trm]) != expected_count:
                    valid = False
                    break
                levels = list(th_nrg[trm])
                b0 = levels[0]
                EXNRG[trm] = [float(x - b0) for x in levels]
            if not valid:
                continue

            case_idx = len(synthetic_cases) + 1
            case_name = f"Synthetic_{case_idx:02d} [{ion} in {sym}, {len(Ar_dict)} params, {','.join(terms)}]"
            synthetic_cases.append({
                "case_name": case_name,
                "is_real": False,
                "ion": ion,
                "symmetry": sym,
                "Ar_dict": Ar_dict,
                "bounds": bounds,
                "constraints": {},
                "EXNRG": EXNRG,
                "term_to_fit": terms,
                "alt_thet_enabled": False,
                "ground_truth_p": p_star.tolist()
            })
            created += 1

    return synthetic_cases


# ==============================================================================
# 4. Worker Evaluation with Iteration Checkpoints
# ==============================================================================
def evaluate_scaling_task(task_args):
    """
    Evaluates configurations on a single instance across niter=60,
    tracking best SSD at iterations 20, 40, and 60.
    """
    sys.stdout = open(os.devnull, 'w')

    case_info, p0, instance_seed = task_args
    bounds = case_info["bounds"]
    constraints = case_info["constraints"]
    EXNRG = copy.deepcopy(case_info["EXNRG"])
    ion = case_info["ion"]
    Ar_dict = case_info["Ar_dict"].copy()
    term_to_fit = case_info["term_to_fit"]
    alt_thet_enabled = case_info["alt_thet_enabled"]

    obj = lambda p: YZS_lib.reconstruct_and_run(
        p, constraints, YZS_lib.core_func, EXNRG, ion, Ar_dict, term_to_fit, alt_thet_enabled
    )

    f0 = obj(p0)

    instance_result = {
        "case_name": case_info["case_name"],
        "is_real": case_info["is_real"],
        "symmetry": case_info["symmetry"],
        "dim": len(bounds),
        "f0": float(f0) if np.isfinite(f0) else None,
        "runs": {}
    }

    if not np.isfinite(f0):
        return instance_result

    for cfg in CONFIGURATIONS:
        cfg_name = cfg["name"]
        ratio = cfg["ratio"]
        mode = cfg["mode"]

        step_taker = BenchmarkBoundedStep(
            bounds=bounds,
            step_ratio=ratio,
            mode=mode,
            seed=instance_seed
        )

        history_ssd = {}
        step_counter = [0]
        current_best = [float(f0)]

        def iteration_callback(x, f, accept):
            step_counter[0] += 1
            if np.isfinite(f) and f < current_best[0]:
                current_best[0] = float(f)
            if step_counter[0] in CHECKPOINTS:
                history_ssd[str(step_counter[0])] = float(current_best[0])
            return False

        t0 = time.time()
        try:
            res = basinhopping(
                obj,
                x0=p0.copy(),
                niter=TOTAL_NITER,
                T=1.0,
                take_step=step_taker,
                callback=iteration_callback,
                seed=instance_seed,
                minimizer_kwargs={
                    'method': 'L-BFGS-B',
                    'jac': '3-point',
                    'bounds': bounds,
                    'options': {'finite_diff_rel_step': 1e-4}
                }
            )
            elapsed = time.time() - t0
            final_f = float(res.fun) if np.isfinite(res.fun) else float(f0)

            running_val = float(f0)
            for cp in CHECKPOINTS:
                cp_str = str(cp)
                if cp_str in history_ssd:
                    running_val = history_ssd[cp_str]
                else:
                    history_ssd[cp_str] = min(running_val, final_f)

            viol_rate = step_taker.boundary_violations / max(1, step_taker.total_steps)

            instance_result["runs"][cfg_name] = {
                "checkpoints": history_ssd,
                "final_ssd": final_f,
                "ssd_drop": float(max(0.0, f0 - final_f)),
                "nfev": int(res.nfev),
                "nit": int(res.nit),
                "time_sec": float(elapsed),
                "violation_rate": float(viol_rate),
                "success": bool(res.lowest_optimization_result.success)
            }
        except Exception as e:
            fallback = {str(cp): float(f0) for cp in CHECKPOINTS}
            instance_result["runs"][cfg_name] = {
                "checkpoints": fallback,
                "final_ssd": float(f0),
                "ssd_drop": 0.0,
                "nfev": 0,
                "nit": 0,
                "time_sec": time.time() - t0,
                "violation_rate": 0.0,
                "error": str(e)
            }

    return instance_result


# ==============================================================================
# 5. Main Execution and Analysis
# ==============================================================================
def main():
    print("================================================================================")
    print("  YouZhiSan Basin-Hopping Iteration Scaling Benchmark (niter = 20, 40, 60)")
    print(f"  CPU Cores: {cpu_count()} | Target Processors: 16 | Checkpoints: {CHECKPOINTS}")
    print("================================================================================\n")

    out_dir = current_dir
    os.makedirs(out_dir, exist_ok=True)

    real_cases = load_real_cases()
    synthetic_cases = generate_synthetic_cases(seed=2026)
    all_cases = real_cases + synthetic_cases
    print(f"Loaded {len(real_cases)} real experimental cases and {len(synthetic_cases)} synthetic cases (Total: {len(all_cases)}).")

    # Generate 3 independent starting points per case
    tasks = []
    random.seed(42)
    np.random.seed(42)

    for case in all_cases:
        bounds = case["bounds"]
        for _ in range(3):
            p0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            instance_seed = 2000 + len(tasks) * 19
            tasks.append((case, p0, instance_seed))

    print(f"Constructed {len(tasks)} paired problem instances ({len(tasks)} x {len(CONFIGURATIONS)} = {len(tasks)*len(CONFIGURATIONS)} basinhopping runs of niter=60).")
    print("Launching parallel worker pool...\n")

    start_wall = time.time()
    n_workers = min(16, cpu_count())
    with Pool(processes=n_workers) as pool:
        all_results = pool.map(evaluate_scaling_task, tasks)
    total_wall = time.time() - start_wall

    print(f"Scaling benchmark completed in {total_wall:.2f} seconds ({total_wall/60.0:.2f} minutes).\n")

    # Save JSON results
    json_path = os.path.join(out_dir, "benchmark_scaling_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw scaling results saved to: {json_path}")

    # ==============================================================================
    # 6. Checkpoint Statistical Compilation
    # ==============================================================================
    cfg_names = [c["name"] for c in CONFIGURATIONS]
    base_name = "10%_clip (Baseline)"

    cp_stats = {
        cp: {name: {"ssd": [], "paired_diff": [], "win": 0, "loss": 0, "tie": 0} for name in cfg_names}
        for cp in CHECKPOINTS
    }

    for inst in all_results:
        runs = inst["runs"]
        if base_name not in runs or "checkpoints" not in runs[base_name]:
            continue

        for cp in CHECKPOINTS:
            cp_str = str(cp)
            base_cp_ssd = runs[base_name]["checkpoints"].get(cp_str, runs[base_name]["final_ssd"])

            for name in cfg_names:
                if name in runs and "checkpoints" in runs[name]:
                    val = runs[name]["checkpoints"].get(cp_str, runs[name]["final_ssd"])
                    cp_stats[cp][name]["ssd"].append(val)
                    diff = val - base_cp_ssd
                    cp_stats[cp][name]["paired_diff"].append(diff)
                    if diff < -1e-4:
                        cp_stats[cp][name]["win"] += 1
                    elif diff > 1e-4:
                        cp_stats[cp][name]["loss"] += 1
                    else:
                        cp_stats[cp][name]["tie"] += 1

    summary_lines = []
    summary_lines.append("===================================================================================================================")
    summary_lines.append("                           BASIN-HOPPING ITERATION SCALING STATISTICAL SUMMARY")
    summary_lines.append(f"  Total Paired Instances: {len(tasks)} | Full Iterations: {TOTAL_NITER} | Wall Time: {total_wall:.2f}s ({total_wall/60:.2f}m)")
    summary_lines.append("===================================================================================================================\n")

    for cp in CHECKPOINTS:
        summary_lines.append(f"--- [Checkpoint: Iteration = {cp}] ---")
        header = f"{'Configuration':<25s} | {'Mean SSD':<10s} | {'Median SSD':<10s} | {'Paired Diff vs Base':<20s} | {'Win / Loss / Tie':<16s}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))

        for name in cfg_names:
            s = cp_stats[cp][name]
            mean_ssd = np.mean(s["ssd"]) if s["ssd"] else 0.0
            med_ssd = np.median(s["ssd"]) if s["ssd"] else 0.0
            diffs = np.array(s["paired_diff"])
            mean_diff = np.mean(diffs) if len(diffs) else 0.0
            std_diff = np.std(diffs) / np.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
            diff_str = f"{mean_diff:+.3f} +/- {std_diff:.3f}" if name != base_name else "0.000 (Baseline)"
            win_str = f"{s['win']:2d} / {s['loss']:2d} / {s['tie']:2d}"
            summary_lines.append(f"{name:<25s} | {mean_ssd:10.3f} | {med_ssd:10.3f} | {diff_str:<20s} | {win_str:<16s}")
        summary_lines.append("\n")

    summary_lines.append("===================================================================================================================\n")
    print("\n" + "\n".join(summary_lines))


if __name__ == '__main__':
    main()
