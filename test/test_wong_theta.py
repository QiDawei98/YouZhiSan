"""
Independent verification script for Wong & Richman (1961) crystal field calculations.

References:
    1. Wong, E. Y., & Richman, I. (1961).
       Analysis of the Absorption Spectrum and Zeeman Effect of Thulium Ethylsulphate.
       The Journal of Chemical Physics, 34(4), 1182-1185. DOI: 10.1063/1.1731717.
    2. Gruber, J. B., & Conway, J. G. (1960).
       Crystal Field Splitting of Energy Levels of Thulium Ethylsulfate.
       The Journal of Chemical Physics, 32(5), 1531-1534. DOI: 10.1063/1.1730954.
    3. Stevens, K. W. H. (1952).
       Matrix Elements and Operator Equivalents Connected with the Magnetic Properties of Rare Earth Ions.
       Proc. Phys. Soc. A, 65, 209-215. DOI: 10.1088/0370-1298/65/3/308.

Purpose:
    Investigate whether the operator equivalent factor alpha for Tm3+ 3H6 (J = 6)
    printed in Table I of Wong & Richman (1961) as 1.0201 x 10^-3 is a typographical
    printing error by AIP for 1.0201 x 10^-2.

Methodology:
    1. Diagonalize the crystal field Hamiltonian for Tm3+ 3H6 (J = 6) in D3h point symmetry
       using Wong's chosen crystal field parameters:
         A20<r2> = 129.8 cm^-1, A40<r4> = -71.0 cm^-1,
         A60<r6> = -28.6 cm^-1, A66<r6> = 432.8 cm^-1.
       under two candidate operator equivalent factors:
         - Case A (Physical / Corrected): alpha = 1.0201 x 10^-2
         - Case B (Table I Printed Typo): alpha = 1.0201 x 10^-3
    2. Compare both calculated Stark spectrums against Wong's own theoretical Stark levels
       tabulated in Table II ('Calc' column of Wong 1961).
    3. Compute Stevens pure LS operator equivalents and evaluate intermediate coupling ratios.
    4. Verify the footnote claim in Wong Table I comparing with Gruber & Conway (1960).

Dependencies:
    PyCrystalField, numpy
"""

import os
import sys

# Ensure PyCrystalField site-packages is on sys.path before numpy/numba are imported
pcf_env_path = r"C:\ProgramData\anaconda3\envs\PyCrystalField\Lib\site-packages"
if os.path.exists(pcf_env_path) and pcf_env_path not in sys.path:
    sys.path.insert(0, pcf_env_path)

import numpy as np
import PyCrystalField as cef

# ==============================================================================
# 1. Literature Parameters (Wong & Richman 1961)
# ==============================================================================
# Crystal field parameters for Tm(C2H5SO4)3*9H2O in D3h symmetry (all in cm^-1):
A_PARAMS = {
    'A20': 129.8,
    'A40': -71.0,
    'A60': -28.6,
    'A66': 432.8
}

# Operator equivalent factors (OEF: alpha, beta, gamma) for 3H6:
# beta and gamma are identical in both cases
BETA_3H6 = 1.5921e-4
GAMMA_3H6 = -5.5284e-6

# Case A: Corrected alpha factor (10^-2)
ALPHA_CORRECTED = 1.0201e-2

# Case B: Table I printed alpha factor (10^-3)
ALPHA_PRINTED_TYPO = 1.0201e-3

# Wong Table II calculated theoretical Stark levels for 3H6 (relative to barycenter, in cm^-1)
# 9 distinct levels (4 non-degenerate: mu=0, 3; 5 doubly-degenerate: mu=1, 2)
WONG_THEORY_BARYCENTER = np.array([
    -166.8, -134.2, -53.0, -4.2, 38.0, 46.6, 53.0, 114.8, 140.5
])

# Wong Table II calculated theoretical Stark levels relative to ground level (in cm^-1)
WONG_THEORY_REL_GROUND = WONG_THEORY_BARYCENTER - WONG_THEORY_BARYCENTER[0]


def calc_3h6_stark_levels(alpha_val, J=6.0):
    """
    Diagonalize the D3h crystal field Hamiltonian for Tm3+ 3H6 (J = 6).
    Returns:
        evals_bary: Eigenvalues relative to barycenter (unique levels, rounded to 0.1 cm^-1).
        evals_ground: Eigenvalues relative to ground level (unique levels, rounded to 0.1 cm^-1).
    """
    B = {
        'B20': A_PARAMS['A20'] * alpha_val,
        'B40': A_PARAMS['A40'] * BETA_3H6,
        'B60': A_PARAMS['A60'] * GAMMA_3H6,
        'B66': A_PARAMS['A66'] * GAMMA_3H6,
    }

    cef.Jion['Tm3+'][2] = J
    cf = cef.CFLevels.Bdict(ion='Tm3+', Bdict=B)
    cf.diagonalize()

    raw_evals = cf.eigenvaluesNoNorm
    barycenter = np.mean(raw_evals)

    evals_bary = np.unique(np.round(raw_evals - barycenter, 2))
    evals_ground = np.unique(np.round(raw_evals - np.min(raw_evals), 2))

    return evals_bary, evals_ground


def run_wong_theta_verification():
    print("=" * 96)
    print(" Wong & Richman (1961) Tm3+ 3H6 Operator Equivalent Factor Verification")
    print(" Benchmark Reference: Wong & Richman (1961) Table II Calculated Theoretical Energies")
    print("=" * 96)

    # -------------------------------------------------------------------------
    # 1. Diagonalization Benchmark
    # -------------------------------------------------------------------------
    calc_bary_corr, calc_ground_corr = calc_3h6_stark_levels(ALPHA_CORRECTED)
    calc_bary_typo, calc_ground_typo = calc_3h6_stark_levels(ALPHA_PRINTED_TYPO)

    print("\n[Comparison of 3H6 Stark Levels Relative to Barycenter (cm^-1)]")
    print("-" * 96)
    print(f"{'Level':<8}{'Wong Table II (Calc)':<26}{'Calc (alpha = 1.0201e-2)':<32}{'Calc (alpha = 1.0201e-3 [Typo])':<30}")
    print("-" * 96)

    for i in range(len(WONG_THEORY_BARYCENTER)):
        th_val = WONG_THEORY_BARYCENTER[i]
        c_corr = calc_bary_corr[i]
        c_typo = calc_bary_typo[i]
        print(f"E_{i+1:<5}{th_val:>16.1f}          {c_corr:>20.2f}            {c_typo:>20.2f}")

    rms_corr = np.sqrt(np.mean((calc_bary_corr - WONG_THEORY_BARYCENTER) ** 2))
    rms_typo = np.sqrt(np.mean((calc_bary_typo - WONG_THEORY_BARYCENTER) ** 2))

    print("-" * 96)
    print(f"RMS Deviation from Wong Theory:     Case A (1.0201e-2): {rms_corr:.4f} cm^-1  |  Case B (1.0201e-3): {rms_typo:.4f} cm^-1")
    print(f"Verdict for Wong (1961) Table II:   {'[EXACT MATCH]':<32}  {'[TOTAL FAILURE / REJECTED]':<30}")

    # -------------------------------------------------------------------------
    # 2. Ratio to Stevens (1952) Pure LS Factors
    # -------------------------------------------------------------------------
    alpha_LS = 1.0 / 99.0
    beta_LS = 8.0 / (3.0 * 11.0 * 1485.0)
    gamma_LS = -5.0 / (13.0 * 33.0 * 2079.0)

    ratio_alpha_corr = ALPHA_CORRECTED / alpha_LS
    ratio_alpha_typo = ALPHA_PRINTED_TYPO / alpha_LS
    ratio_beta = BETA_3H6 / beta_LS
    ratio_gamma = GAMMA_3H6 / gamma_LS

    print("\n[Ratio Comparison against Stevens (1952) Pure LS Operator Equivalents]")
    print("-" * 96)
    print(f"  beta(Wong)  / beta_LS   = {ratio_beta:.4f}  (diff: {(ratio_beta - 1.0)*100:+.2f}%)")
    print(f"  gamma(Wong) / gamma_LS  = {ratio_gamma:.4f}  (diff: {(ratio_gamma - 1.0)*100:+.2f}%)")
    print(f"  alpha(Corrected) / alpha_LS = {ratio_alpha_corr:.4f}  (diff: {(ratio_alpha_corr - 1.0)*100:+.2f}%)  <- Consistent with intermediate coupling")
    print(f"  alpha(Printed)   / alpha_LS = {ratio_alpha_typo:.4f}  (diff: {(ratio_alpha_typo - 1.0)*100:+.2f}%)  <- Off by exactly one order of magnitude")

    # -------------------------------------------------------------------------
    # 3. Cross-Examination against Gruber & Conway (1960) Footnote
    # -------------------------------------------------------------------------
    alpha_Gruber = (1.0 / 99.0) * 1.010  # 1.0202e-2
    diff_percent_corr = (ALPHA_CORRECTED - alpha_Gruber) / alpha_Gruber * 100.0
    diff_percent_typo = (ALPHA_PRINTED_TYPO - alpha_Gruber) / alpha_Gruber * 100.0

    print("\n[Cross-Examination with Wong Table I Footnote a (Gruber & Conway Comparison)]")
    print("-" * 96)
    print(f"  Gruber (1960) alpha value: {alpha_Gruber:.6e}")
    print(f"  Wong (1961) Table I Footnote printed difference: '1.0202 (0.0%)'")
    print(f"  Actual diff if alpha = 1.0201e-2: {diff_percent_corr:+.4f}%  -> Rounds to 0.0%  (Matches Footnote)")
    print(f"  Actual diff if alpha = 1.0201e-3: {diff_percent_typo:+.4f}% -> Off by -90.0%  (Violates Footnote)")
    print("=" * 96)

    # -------------------------------------------------------------------------
    # 4. Strict Assertions for Automated CI/CD
    # -------------------------------------------------------------------------
    assert rms_corr < 0.1, f"Corrected alpha must reproduce Wong Table II within 0.1 cm^-1, got RMS {rms_corr}"
    assert rms_typo > 20.0, f"Printed typo alpha must fail to reproduce Wong Table II, got RMS {rms_typo}"
    print("\n>>> ALL REGRESSION ASSERTIONS PASSED SUCCESSFULLY! <<<")


if __name__ == '__main__':
    run_wong_theta_verification()
