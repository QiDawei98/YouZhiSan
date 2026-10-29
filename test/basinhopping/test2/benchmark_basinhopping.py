"""
Basin-Hopping Step Size & Boundary Handling Benchmark for YouZhiSan.
Author: Qi Dawei / AI Assistant
Description:
    Evaluates step size ratios (10%, 15%, 20%, 25%, 30%) and boundary handling
    (clipping vs vectorized mirror reflection) on the basinhopping optimization stage
    across crystal field Hamiltonians.
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
if os.path.exists(site_pkg) and site_pkg not in sys.path:
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
# 1. Custom Step Operator (BoundedStep)
# ==============================================================================
class BenchmarkBoundedStep:
    """
    Uniform boundary-scaled perturbation operator with configurable mode (clip vs reflect).
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
            # Safety clipping against floating-point rounding
            np.clip(x_new, low, high, out=x_new)
        else:
            # Absorption / Clipping
            np.clip(x_new, low, high, out=x_new)

        return x_new


# ==============================================================================
# 2. Configurations to Test (10 Configurations)
# ==============================================================================
CONFIGURATIONS = [
    {"name": "10%_clip (Baseline)", "ratio": 0.10, "mode": "clip"},
    {"name": "10%_reflect",          "ratio": 0.10, "mode": "reflect"},
    {"name": "15%_clip",             "ratio": 0.15, "mode": "clip"},
    {"name": "15%_reflect",          "ratio": 0.15, "mode": "reflect"},
    {"name": "20%_clip",             "ratio": 0.20, "mode": "clip"},
    {"name": "20%_reflect (Target)", "ratio": 0.20, "mode": "reflect"},
    {"name": "25%_clip",             "ratio": 0.25, "mode": "clip"},
    {"name": "25%_reflect",          "ratio": 0.25, "mode": "reflect"},
    {"name": "30%_clip",             "ratio": 0.30, "mode": "clip"},
    {"name": "30%_reflect",          "ratio": 0.30, "mode": "reflect"},
]


# ==============================================================================
# 3. Real and Synthetic Case Generation
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
# 4. Worker Evaluation on a Single Paired Instance
# ==============================================================================
def evaluate_instance_task(task_args):
    """
    Evaluates all 10 configurations on a single (case, initial_p0, base_seed, trial_idx) tuple.
    """
    sys.stdout = open(os.devnull, 'w')

    case_info, p0, instance_seed, trial_idx = task_args
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
        "ion": ion,
        "trial_idx": trial_idx,
        "dim": len(bounds),
        "f0": float(f0) if np.isfinite(f0) else None,
        "runs": {}
    }

    if not np.isfinite(f0):
        return instance_result

    # Evaluate each configuration with strictly paired seeds
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

        t0 = time.time()
        try:
            res = basinhopping(
                obj,
                x0=p0.copy(),
                niter=25,
                T=1.0,
                take_step=step_taker,
                seed=instance_seed,
                minimizer_kwargs={
                    'method': 'L-BFGS-B',
                    'jac': '3-point',
                    'bounds': bounds,
                    'options': {'finite_diff_rel_step': 1e-4}
                }
            )
            elapsed = time.time() - t0
            final_ssd = float(res.fun) if np.isfinite(res.fun) else float(f0)
            viol_rate = step_taker.boundary_violations / max(1, step_taker.total_steps)

            # Reconstruct parameter dictionaries
            p_opt = res.x
            total_vars = len(p_opt) + len(constraints)
            p_full = np.zeros(total_vars)
            ind_indices = sorted(list(set(range(total_vars)) - set(constraints.keys())))
            p_full[ind_indices] = p_opt
            for dep, (ind, ratio_val) in constraints.items():
                p_full[dep] = p_full[ind] * ratio_val

            all_keys = list(Ar_dict.keys())
            full_cfp = {k: float(round(p_full[i], 6)) for i, k in enumerate(all_keys)}
            ind_keys = [all_keys[i] for i in ind_indices]
            ind_cfp = {k: float(round(p_opt[i], 6)) for i, k in enumerate(ind_keys)}

            instance_result["runs"][cfg_name] = {
                "final_ssd": final_ssd,
                "ssd_drop": float(max(0.0, f0 - final_ssd)),
                "nfev": int(res.nfev),
                "nit": int(res.nit),
                "time_sec": float(elapsed),
                "violation_rate": float(viol_rate),
                "success": bool(res.lowest_optimization_result.success),
                "independent_cfp": ind_cfp,
                "full_cfp": full_cfp
            }
        except Exception as e:
            instance_result["runs"][cfg_name] = {
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
# 5. Main Orchestration
# ==============================================================================
def main():
    print("================================================================================")
    print("  YouZhiSan Basin-Hopping Step Size & Boundary Handling Benchmark")
    print(f"  CPU Cores: {cpu_count()} | Target Processors: 16 | Target Stage: basinhopping only")
    print("================================================================================\n")

    out_dir = current_dir
    os.makedirs(out_dir, exist_ok=True)

    real_cases = load_real_cases()
    synthetic_cases = generate_synthetic_cases(seed=2026)
    all_cases = real_cases + synthetic_cases
    print(f"Loaded {len(real_cases)} real experimental cases and {len(synthetic_cases)} synthetic cases.")
    print(f"Total cases: {len(all_cases)}\n")

    # Generate 3 independent starting points per case
    tasks = []
    random.seed(42)
    np.random.seed(42)

    for case in all_cases:
        bounds = case["bounds"]
        for trial_idx in range(3):
            p0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            instance_seed = 1000 + len(tasks) * 17
            tasks.append((case, p0, instance_seed, trial_idx + 1))

    print(f"Constructed {len(tasks)} paired problem instances ({len(tasks)} x {len(CONFIGURATIONS)} = {len(tasks)*len(CONFIGURATIONS)} total basinhopping runs).")
    print("Launching parallel worker pool...\n")

    start_wall = time.time()
    n_workers = min(16, cpu_count())
    with Pool(processes=n_workers) as pool:
        all_results = pool.map(evaluate_instance_task, tasks)
    total_wall = time.time() - start_wall

    print(f"Parallel benchmarking completed in {total_wall:.2f} seconds ({total_wall/60.0:.2f} minutes).\n")

    # Save raw benchmark JSON
    json_path = os.path.join(out_dir, "benchmark_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw results saved to: {json_path}")

    # Export dedicated fitted CFP parameters for 10%_reflect
    target_cfg = "10%_reflect"
    cfp_export = []
    for idx, inst in enumerate(all_results):
        if target_cfg in inst["runs"]:
            r = inst["runs"][target_cfg]
            cfp_export.append({
                "task_idx": idx,
                "case_name": inst["case_name"],
                "is_real": inst["is_real"],
                "symmetry": inst["symmetry"],
                "ion": inst.get("ion", ""),
                "trial_idx": inst.get("trial_idx", 1),
                "dim": inst["dim"],
                "f0": inst["f0"],
                "final_ssd": r["final_ssd"],
                "time_sec": r["time_sec"],
                "nfev": r["nfev"],
                "nit": r["nit"],
                "success": r["success"],
                "independent_cfp": r.get("independent_cfp", {}),
                "full_cfp": r.get("full_cfp", {})
            })
    cfp_path = os.path.join(out_dir, "fitted_cfp_10pct_reflect.json")
    with open(cfp_path, 'w', encoding='utf-8') as f:
        json.dump(cfp_export, f, indent=2)
    print(f"Fitted CFP parameters for 10%_reflect saved to: {cfp_path}")

    # ==============================================================================
    # 6. Statistical Analysis & Paired Comparisons
    # ==============================================================================
    cfg_names = [c["name"] for c in CONFIGURATIONS]
    baseline_name = "10%_clip (Baseline)"

    stats = {
        name: {
            "final_ssd": [], "time": [], "nfev": [], "viol_rate": [],
            "diff_vs_baseline": [], "win_count": 0, "loss_count": 0, "tie_count": 0
        }
        for name in cfg_names
    }

    for inst in all_results:
        runs = inst["runs"]
        if baseline_name not in runs or "final_ssd" not in runs[baseline_name]:
            continue
        base_ssd = runs[baseline_name]["final_ssd"]

        for name in cfg_names:
            if name in runs and "final_ssd" in runs[name]:
                val = runs[name]["final_ssd"]
                stats[name]["final_ssd"].append(val)
                stats[name]["time"].append(runs[name]["time_sec"])
                stats[name]["nfev"].append(runs[name]["nfev"])
                stats[name]["viol_rate"].append(runs[name]["violation_rate"])

                diff = val - base_ssd
                stats[name]["diff_vs_baseline"].append(diff)
                if diff < -1e-4:
                    stats[name]["win_count"] += 1
                elif diff > 1e-4:
                    stats[name]["loss_count"] += 1
                else:
                    stats[name]["tie_count"] += 1

    summary_lines = []
    summary_lines.append("===================================================================================================================")
    summary_lines.append("                               BASIN-HOPPING BENCHMARK STATISTICAL SUMMARY")
    summary_lines.append(f"  Total Paired Instances: {len(tasks)} | Iterations per Run: 25 | Total Wall Time: {total_wall:.2f}s")
    summary_lines.append("===================================================================================================================")
    header = f"{'Configuration':<25s} | {'Mean SSD':<10s} | {'Median SSD':<10s} | {'Paired Diff vs Base':<20s} | {'Win / Loss / Tie':<16s} | {'Viol%':<6s} | {'Avg Time':<8s}"
    summary_lines.append(header)
    summary_lines.append("-" * len(header))

    for name in cfg_names:
        s = stats[name]
        mean_ssd = np.mean(s["final_ssd"]) if s["final_ssd"] else 0.0
        med_ssd = np.median(s["final_ssd"]) if s["final_ssd"] else 0.0

        diffs = np.array(s["diff_vs_baseline"])
        mean_diff = np.mean(diffs) if len(diffs) else 0.0
        std_diff = np.std(diffs) / np.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
        diff_str = f"{mean_diff:+.3f} +/- {std_diff:.3f}" if name != baseline_name else "0.000 (Baseline)"

        win_str = f"{s['win_count']:2d} / {s['loss_count']:2d} / {s['tie_count']:2d}"
        viol_pct = np.mean(s["viol_rate"]) * 100 if s["viol_rate"] else 0.0
        avg_time = np.mean(s["time"]) if s["time"] else 0.0

        summary_lines.append(f"{name:<25s} | {mean_ssd:10.3f} | {med_ssd:10.3f} | {diff_str:<20s} | {win_str:<16s} | {viol_pct:5.1f}% | {avg_time:6.2f}s")

    summary_lines.append("===================================================================================================================\n")
    print("\n" + "\n".join(summary_lines))


if __name__ == '__main__':
    main()
