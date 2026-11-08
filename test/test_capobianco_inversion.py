import numpy as np
import PyCrystalField as cef

# Er3+:YVO4 crystal field parameters (Capobianco et al., 1997, in meV)
AR = {'AR20': -13.393393, 'AR40': 3.040248, 'AR44': 117.704110, 'AR60': -5.284904, 'AR64': 4.370623}

# Stevens factors from Erath (1961) Table I
ROW4 = [-0.0335e-3, 1.7901e-4, 3.7779e-5]   # Printed label '4F9/2' (12497 cm^-1)
ROW5 = [-5.5440e-3, 1.9305e-4, -8.8316e-6]  # Printed label '4I9/2' (15447 cm^-1)

def calc_levels(oef):
    B = {
        'B20': AR['AR20'] * oef[0],
        'B40': AR['AR40'] * oef[1],
        'B44': AR['AR44'] * oef[1],
        'B60': AR['AR60'] * oef[2],
        'B64': AR['AR64'] * oef[2],
    }
    cef.Jion['Er3+'][2] = 4.5
    cf = cef.CFLevels.Bdict(ion='Er3+', Bdict=B)
    cf.diagonalize()
    evals = np.unique(np.round(cf.eigenvalues, 4))
    evals.sort()
    return evals - evals[0]

# Experimental data (Capobianco et al., 1997)
exp_data = {
    '4I9/2': [0.0, None, None, None, 28.1444],
    '4F9/2': [0.0, 3.3476, 9.0508, 14.3822, 16.8619]
}

lev_row4 = calc_levels(ROW4)
lev_row5 = calc_levels(ROW5)

cases = {
    'Scheme A': {'4I9/2': lev_row5, '4F9/2': lev_row4},
    'Scheme B': {'4I9/2': lev_row4, '4F9/2': lev_row5},
}

def run_test():
    print("=" * 64)
    print("Configuration Setup:")
    print("  Scheme A: Table I printed labels (4I9/2 <- Row 5, 4F9/2 <- Row 4)")
    print("  Scheme B: Inverted assignment    (4I9/2 <- Row 4, 4F9/2 <- Row 5)")
    print("=" * 64)
    print(f"{'Multiplet':<10}{'Level':<6}{'Exp (meV)':<14}{'Scheme A':<16}{'Scheme B':<16}")
    print("-" * 64)

    for term in ['4I9/2', '4F9/2']:
        for i in range(5):
            e_exp = f"{exp_data[term][i]:.4f}" if exp_data[term][i] is not None else "-"
            e_a = f"{cases['Scheme A'][term][i]:.4f}"
            e_b = f"{cases['Scheme B'][term][i]:.4f}"
            lbl = term if i == 0 else ""
            print(f"{lbl:<10}E_{i+1:<4}{e_exp:<14}{e_a:<16}{e_b:<16}")
        print("-" * 64)

    def get_rms(case_dict):
        diffs = []
        for term, levels in exp_data.items():
            for i, val in enumerate(levels):
                if val is not None:
                    diffs.append(case_dict[term][i] - val)
        return np.sqrt(np.mean(np.array(diffs) ** 2))

    rms_a = get_rms(cases['Scheme A'])
    rms_b = get_rms(cases['Scheme B'])
    print(f"{'RMS Error (meV)':<20}{rms_a:>14.4f}{rms_b:>16.4f}")
    print("=" * 64)

if __name__ == '__main__':
    run_test()
