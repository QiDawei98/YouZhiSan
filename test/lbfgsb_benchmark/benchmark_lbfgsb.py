"""
Comprehensive Benchmark for L-BFGS-B Parameter Configurations in YouZhiSan.
Author: Qi Dawei / AI Assistant
Environment: PyCrystalField (C:\\ProgramData\\anaconda3\\envs\\PyCrystalField\\python.exe)
Purpose: Evaluate optimizer convergence, gradient step behavior, SSD reduction,
         speed, and stability across real experimental datasets and physically
         valid synthetic crystal field Hamiltonians.

Strictly preserves SciPy's natural convergence limits (NO maxiter truncation).
Uses adaptive load balancing: 1 trial for triclinic/monoclinic (10-14 parameters),
3 trials for medium (7-9 parameters), 6 trials for high symmetry (4-6 parameters).
"""

import os
import sys
import re
import copy
import time
import json
import random
import argparse
from multiprocessing import Pool, cpu_count

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
from scipy.optimize import minimize
import YZS_lib

# ==============================================================================
# 1. Parameter Configurations to Test (Natural SciPy Stopping, No maxiter)
# ==============================================================================
CONFIGURATIONS = {
    # Baseline
    "1. Default (eps=1e-8, 2-pt)": {
        "kwargs": {
            "method": "L-BFGS-B",
            "options": {"eps": 1e-8}
        }
    },
    # Option B family (2-point forward difference with reasonable step)
    "2. Option B (eps=1e-5, 2-pt)": {
        "kwargs": {
            "method": "L-BFGS-B",
            "options": {"eps": 1e-5}
        }
    },
    "3. Option B tune (eps=1e-4, 2-pt)": {
        "kwargs": {
            "method": "L-BFGS-B",
            "options": {"eps": 1e-4}
        }
    },
    # Option A family (3-point central difference)
    "4. Option A (rel=1e-4, 3-pt) [Current Repo]": {
        "kwargs": {
            "method": "L-BFGS-B",
            "jac": "3-point",
            "options": {"finite_diff_rel_step": 1e-4}
        }
    },
    "5. Option A tune tight (rel=5e-5, 3-pt)": {
        "kwargs": {
            "method": "L-BFGS-B",
            "jac": "3-point",
            "options": {"finite_diff_rel_step": 5e-5}
        }
    },
    "6. Option A tune loose (rel=5e-4, 3-pt)": {
        "kwargs": {
            "method": "L-BFGS-B",
            "jac": "3-point",
            "options": {"finite_diff_rel_step": 5e-4}
        }
    },
    # Previous AI proposals
    "7. Model 2 R2 (rel=1e-3, 3-pt)": {
        "kwargs": {
            "method": "L-BFGS-B",
            "jac": "3-point",
            "options": {"finite_diff_rel_step": 1e-3}
        }
    },
    "8. Model 1 (eps=1e-2, 2-pt, loose tol)": {
        "kwargs": {
            "method": "L-BFGS-B",
            "options": {"eps": 1e-2, "ftol": 1e-7, "gtol": 1e-5}
        }
    }
}

# ==============================================================================
# 2. Real Input File Parser
# ==============================================================================
def parse_real_case(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    processed_lines = []
    for line in lines:
        line_no_comment = re.sub(r'#.*', '', line)
        if line_no_comment.strip() and re.search(r'#.*', line):
            processed_lines.append(line_no_comment)
        elif not re.search(r'#.*', line):
            processed_lines.append(line_no_comment)
    lines = [l.upper() for l in processed_lines]

    if sum(1 for l in lines if re.fullmatch(r'\*+', l.strip())) != 1:
        return None

    in_block = False
    Ar_dict = {}
    individual_bounds = []
    shared_bounds = None
    symmetry = None
    alt_thet_enabled = False
    ion = None

    for line in lines:
        if in_block:
            if m := re.fullmatch(r'\s{4}(AR\d-?\d)', line.rstrip()):
                Ar_dict[m.group(1)] = None
                individual_bounds.append(None)
                continue
            elif m := re.fullmatch(r'\s{4}(AR\d-?\d)\s+\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)', line.rstrip()):
                Ar_dict[m.group(1)] = None
                individual_bounds.append((float(m.group(2)), float(m.group(3))))
                continue
            else:
                in_block = False

        if re.fullmatch(r'\*+', line.rstrip()):
            break
        elif m := re.fullmatch(r'ION\s+(.+)', line.rstrip()):
            ion = m.group(1).strip()
        elif m := re.fullmatch(r'SYMMETRY\s+(.+)', line.rstrip()):
            symmetry = m.group(1).strip()
        elif re.fullmatch(r'PARAMETERS', line.rstrip()):
            in_block = True
        elif m := re.fullmatch(r'BOUNDS\s+\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)', line.rstrip()):
            shared_bounds = (float(m.group(1)), float(m.group(2)))
        elif m := re.fullmatch(r'ALT_THET\s+(TRUE|FALSE)', line.rstrip()):
            alt_thet_enabled = (m.group(1) == 'TRUE')

    if not ion:
        return None
    if not Ar_dict and symmetry:
        Ar_dict = YZS_lib.allowed_cfp(symmetry)
        individual_bounds = [None] * len(Ar_dict)
    if not Ar_dict:
        return None

    for i in range(len(individual_bounds)):
        if individual_bounds[i] is None:
            individual_bounds[i] = shared_bounds if shared_bounds else (-250.0, 250.0)

    EXNRG = {}
    in_block = False
    start = False
    block_name = None
    for line in lines:
        if re.fullmatch(r'\*+', line.strip()):
            start = True
            continue
        if start:
            if in_block:
                if re.fullmatch(r'-?\d+(\.\d+)?', line.rstrip()):
                    EXNRG[block_name].append(float(line.rstrip()))
                    continue
                elif re.fullmatch(r'NONE', line.rstrip()):
                    EXNRG[block_name].append(None)
                    continue
                elif line.isspace():
                    in_block = False
                    continue
            if line.isspace():
                continue
            elif re.fullmatch(r'\d+[A-Z](\d+\/\d+|\d+)', line.rstrip()):
                in_block = True
                block_name = line.rstrip()
                EXNRG[block_name] = []
                continue

    # Subtract baseline
    for k, v in EXNRG.items():
        if all(isinstance(x, float) for x in v):
            b0 = v[0]
            for i in range(len(v)):
                v[i] = v[i] - b0
        else:
            b0 = 0.0
            for x in v:
                if isinstance(x, float):
                    b0 = x
                    break
            for i in range(len(v)):
                if isinstance(v[i], float):
                    v[i] = v[i] - b0

    thet_dict = YZS_lib.alt_Thet if alt_thet_enabled else YZS_lib.Thet
    term_to_fit = [IT[1] for IT in thet_dict if IT[0].upper() == ion and IT[1] in EXNRG]
    if not term_to_fit:
        return None

    # For massive systems with 10+ terms (e.g. Er3+ in C1h), cap to 4 most significant terms for benchmarking
    if len(term_to_fit) > 4:
        term_to_fit = term_to_fit[:4]
        EXNRG = {k: EXNRG[k] for k in term_to_fit}

    if symmetry in ['OH', 'O', 'TD']:
        constraints = {1: (0, 5), 3: (2, -21)}
    elif symmetry in ['T', 'TH']:
        constraints = {1: (0, 5), 3: (2, -21), 5: (4, -1)}
    else:
        constraints = {}

    indices_to_remove = set(constraints.keys())
    filtered_bounds = [b for i, b in enumerate(individual_bounds) if i not in indices_to_remove]

    case_name = os.path.basename(os.path.dirname(file_path)) + " / " + os.path.basename(file_path)

    return {
        "case_name": case_name,
        "is_real": True,
        "ion": ion,
        "symmetry": symmetry,
        "Ar_dict": Ar_dict,
        "bounds": filtered_bounds,
        "constraints": constraints,
        "EXNRG": EXNRG,
        "term_to_fit": term_to_fit,
        "alt_thet_enabled": alt_thet_enabled
    }

# ==============================================================================
# 3. Synthetic Case Generator (100 Physics-Consistent Fictional Cases)
# ==============================================================================
def generate_synthetic_cases(count=100, seed=2026):
    random.seed(seed)
    np.random.seed(seed)

    non_cubic = [s for s in YZS_lib.allowed_symmetries if s not in YZS_lib.cubic_symmetry]
    available_ions = sorted(list(set(k[0] for k in YZS_lib.Thet.keys())))

    synthetic_cases = []
    attempts = 0
    max_attempts = count * 50

    while len(synthetic_cases) < count and attempts < max_attempts:
        attempts += 1
        sym = random.choice(non_cubic)
        ion = random.choice(available_ions)

        valid_terms = [k[1] for k in YZS_lib.Thet.keys() if k[0] == ion]
        if not valid_terms:
            continue

        # Use 1 or 2 spectral terms
        k_terms = 1 if len(valid_terms) == 1 or random.random() < 0.65 else 2
        terms = random.sample(valid_terms, k_terms)

        Ar_dict = YZS_lib.allowed_cfp(sym)
        n_params = len(Ar_dict)
        if n_params == 0:
            continue

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
            norm_levels = [float(x - b0) for x in levels]
            EXNRG[trm] = norm_levels

        if not valid:
            continue

        case_idx = len(synthetic_cases) + 1
        case_name = f"Synthetic_{case_idx:03d} [{ion} in {sym}, {','.join(terms)}]"

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

    return synthetic_cases

# ==============================================================================
# 4. Worker Execution Task
# ==============================================================================
def run_single_instance(task_args):
    """
    Evaluates all 8 configurations on a single problem instance (case + start_point).
    """
    sys.stdout = open(os.devnull, 'w')

    case_info, p0 = task_args
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
    results = {
        "case_name": case_info["case_name"],
        "is_real": case_info["is_real"],
        "initial_ssd": float(f0) if np.isfinite(f0) else None,
        "config_results": {}
    }

    if not np.isfinite(f0):
        return results

    for cfg_name, cfg in CONFIGURATIONS.items():
        t0 = time.time()
        try:
            res = minimize(
                obj,
                p0.copy(),
                bounds=bounds,
                **cfg["kwargs"]
            )
            elapsed = time.time() - t0
            f_val = float(res.fun) if np.isfinite(res.fun) else 1e12

            premature = bool(res.nit <= 1 and (f0 - f_val) < 1e-4)

            results["config_results"][cfg_name] = {
                "final_ssd": float(f_val),
                "nit": int(res.nit),
                "nfev": int(res.nfev),
                "time_sec": float(elapsed),
                "success": bool(res.success),
                "premature_stop": bool(premature),
                "ssd_drop": float(max(0.0, f0 - f_val))
            }
        except Exception as e:
            results["config_results"][cfg_name] = {
                "final_ssd": float(f0),
                "nit": 0,
                "nfev": 0,
                "time_sec": time.time() - t0,
                "success": False,
                "premature_stop": True,
                "ssd_drop": 0.0,
                "error": str(e)
            }

    return results

# ==============================================================================
# 5. Master Benchmark Orchestration
# ==============================================================================
def run_benchmark():
    t_start = time.time()
    num_cpus = cpu_count()
    print("=" * 80, flush=True)
    print("  YOUZHISAN L-BFGS-B ROBUSTNESS & CONVERGENCE BENCHMARK", flush=True)
    print(f"  CPU Cores Detected: {num_cpus} (Pool size set to {num_cpus})", flush=True)
    print("=" * 80, flush=True)

    # 1. Collect real cases (test_dir is the parent 'test' folder)
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.dirname(benchmark_dir)
    txt_files = sorted([
        os.path.join(r, f)
        for r, _, files in os.walk(test_dir)
        if not r.startswith(benchmark_dir)
        for f in files if f.endswith('.txt') and 'res' not in f.lower()
    ])

    real_cases = []
    for f in txt_files:
        rc = parse_real_case(f)
        if rc is not None:
            real_cases.append(rc)

    print(f"Loaded {len(real_cases)} real experimental cases.", flush=True)

    # 2. Generate 100 synthetic cases
    n_synthetic = 100
    print(f"Generating {n_synthetic} physics-grounded non-cubic synthetic cases...", flush=True)
    synthetic_cases = generate_synthetic_cases(count=n_synthetic, seed=2026)
    print(f"Generated {len(synthetic_cases)} synthetic cases.", flush=True)

    all_cases = real_cases + synthetic_cases
    print(f"Total cases: {len(all_cases)} (Real: {len(real_cases)}, Synthetic: {len(synthetic_cases)})", flush=True)

    # 3. Adaptive Load Balancing for low symmetry (单斜/三斜)
    # High symmetry (params <= 6): 6 trials per case
    # Medium symmetry (params 7-9): 3 trials per case
    # Low symmetry (monoclinic/triclinic, params >= 10): 1 trial per case
    np.random.seed(42)
    tasks = []
    for case in all_cases:
        bounds = case["bounds"]
        n_params = len(bounds)
        lowers = [b[0] for b in bounds]
        uppers = [b[1] for b in bounds]

        if n_params <= 6:
            trials = 6
        elif n_params <= 9:
            trials = 3
        else:
            trials = 1  # 1 trial for monoclinic/triclinic

        valid_starts = 0
        attempts = 0
        while valid_starts < trials and attempts < trials * 25:
            attempts += 1
            p0 = np.array([np.random.uniform(l, u) for l, u in zip(lowers, uppers)])
            f0 = YZS_lib.reconstruct_and_run(
                p0, case["constraints"], YZS_lib.core_func,
                copy.deepcopy(case["EXNRG"]), case["ion"],
                case["Ar_dict"].copy(), case["term_to_fit"], case["alt_thet_enabled"]
            )
            if np.isfinite(f0):
                weight = n_params * len(case["term_to_fit"])
                tasks.append(((case, p0), weight))
                valid_starts += 1

    # Sort tasks in descending order of weight (Longest-Processing-Time-First)
    tasks.sort(key=lambda x: x[1], reverse=True)
    work_tasks = [t[0] for t in tasks]

    print(f"Total task instances to evaluate: {len(work_tasks)}", flush=True)
    print(f"Total L-BFGS-B optimization runs: {len(work_tasks) * len(CONFIGURATIONS)}", flush=True)
    print(f"Dispatching tasks across {num_cpus} parallel processes (chunksize=1, dynamic scheduling)...", flush=True)

    # 4. Execute parallel pool with chunksize=1 for dynamic load balancing
    results = []
    with Pool(processes=num_cpus) as pool:
        total = len(work_tasks)
        completed = 0
        last_pct = -1
        t0_pool = time.time()

        for res in pool.imap_unordered(run_single_instance, work_tasks, chunksize=1):
            results.append(res)
            completed += 1
            pct = int(completed / total * 100)
            if pct != last_pct and (pct % 5 == 0 or completed == total):
                last_pct = pct
                el = time.time() - t0_pool
                eta = (el / completed) * (total - completed) if completed > 0 else 0
                print(f"  [Progress: {pct:3d}%] {completed}/{total} instances finished. Elapsed: {el/60:.2f}m, ETA: {eta/60:.2f}m", flush=True)

    # 5. Analyze Results
    print("\n" + "=" * 80, flush=True)
    print("  COMPUTING AGGREGATE PERFORMANCE METRICS", flush=True)
    print("=" * 80, flush=True)

    summary_text = analyze_and_print_results(results)

    # 6. Save results to disk
    json_path = os.path.join(benchmark_dir, "benchmark_lbfgsb_results.json")
    txt_path = os.path.join(benchmark_dir, "benchmark_lbfgsb_summary.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    compact_records = []
    for r in results:
        compact_records.append({
            "case": r["case_name"],
            "is_real": r["is_real"],
            "initial_ssd": r["initial_ssd"],
            "configs": r["config_results"]
        })
    def json_default(o):
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        return str(o)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(compact_records, f, indent=2, default=json_default)

    total_time = time.time() - t_start
    print(f"\nAll benchmark tasks completed in {total_time/60:.2f} minutes.", flush=True)
    print(f"Summary report saved to: {txt_path}", flush=True)
    print(f"Detailed raw JSON saved to: {json_path}", flush=True)

def analyze_and_print_results(results):
    lines = []
    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    valid_res = [r for r in results if r["initial_ssd"] is not None]
    real_res = [r for r in valid_res if r["is_real"]]
    synth_res = [r for r in valid_res if not r["is_real"]]

    log(f"Total Valid Tested Instances: {len(valid_res)} (Real: {len(real_res)}, Synthetic: {len(synth_res)})")

    def evaluate_group(group_name, group_data):
        log("\n" + "-" * 80)
        log(f"  METRICS REPORT: {group_name} (Sample Count: {len(group_data)})")
        log("-" * 80)
        log(f"{'Configuration':<45} | {'Mean SSD':<12} | {'Premature%':<11} | {'Mean nfev':<10} | {'Time/run(s)':<11} | {'Score'}")
        log("-" * 105)

        cfg_stats = {}
        for cfg_name in CONFIGURATIONS.keys():
            ssds = []
            prematures = []
            nfevs = []
            times = []
            ssd_drops = []

            for item in group_data:
                c = item["config_results"].get(cfg_name)
                if not c:
                    continue
                ssds.append(c["final_ssd"])
                prematures.append(1.0 if c["premature_stop"] else 0.0)
                nfevs.append(c["nfev"])
                times.append(c["time_sec"])
                ssd_drops.append(c["ssd_drop"])

            if not ssds:
                continue

            mean_ssd = np.mean(ssds)
            median_ssd = np.median(ssds)
            premature_pct = np.mean(prematures) * 100
            mean_nfev = np.mean(nfevs)
            mean_time = np.mean(times)
            mean_drop = np.mean(ssd_drops)

            cfg_stats[cfg_name] = {
                "mean_ssd": mean_ssd,
                "median_ssd": median_ssd,
                "premature_pct": premature_pct,
                "mean_nfev": mean_nfev,
                "mean_time": mean_time,
                "mean_drop": mean_drop,
            }

        sorted_by_ssd = sorted(cfg_stats.keys(), key=lambda k: cfg_stats[k]["mean_ssd"])
        sorted_by_prem = sorted(cfg_stats.keys(), key=lambda k: cfg_stats[k]["premature_pct"])
        sorted_by_time = sorted(cfg_stats.keys(), key=lambda k: cfg_stats[k]["mean_time"])

        ranks = {}
        for k in cfg_stats.keys():
            ranks[k] = sorted_by_ssd.index(k) + 2.0 * sorted_by_prem.index(k) + 0.5 * sorted_by_time.index(k)

        best_cfg = min(ranks.keys(), key=lambda k: ranks[k])

        for k in sorted(cfg_stats.keys(), key=lambda x: ranks[x]):
            s = cfg_stats[k]
            mark = " [BEST]" if k == best_cfg else ""
            log(f"{k:<45} | {s['mean_ssd']:<12.3f} | {s['premature_pct']:<10.1f}% | {s['mean_nfev']:<10.1f} | {s['mean_time']:<11.3f} | {ranks[k]:.1f}{mark}")

        log(f"\n>>> Best Configuration for {group_name}: {best_cfg}")
        return best_cfg, cfg_stats

    best_real, stats_real = evaluate_group("REAL EXPERIMENTAL CASES", real_res)
    best_synth, stats_synth = evaluate_group("SYNTHETIC CASES", synth_res)
    best_overall, stats_all = evaluate_group("OVERALL (REAL + SYNTHETIC)", valid_res)

    log("\n" + "=" * 80)
    log("  FINAL CONCLUSION & RECOMMENDATION")
    log("=" * 80)
    log(f"1. Best L-BFGS-B Configuration on REAL Data    : {best_real}")
    log(f"2. Best L-BFGS-B Configuration on SYNTHETIC Data: {best_synth}")
    log(f"3. Overall Best Configuration (Global Combined) : {best_overall}")
    log("=" * 80)

    return "\n".join(lines)

if __name__ == "__main__":
    run_benchmark()
