"""
Regression test for get_stark_levels_count in YZS_lib.py.
Verifies Stark-level counts across all 32 crystallographic point groups
against the group-theoretical values tabulated in Chen Xueyuan (1998) Table 2.1.
"""

import sys
import os

# Add parent directory to sys.path to import YZS_lib
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from YZS_lib import get_stark_levels_count, allowed_symmetries

# Reference values from Chen Xueyuan (1998) Dissertation, Table 2.1
EXPECTED_INTEGER_COUNTS = {
    'CUBIC': {0: 1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 6, 7: 6, 8: 7},
    'HEXAGONAL': {0: 1, 1: 2, 2: 3, 3: 5, 4: 6, 5: 7, 6: 9, 7: 10, 8: 11},
    'TRIGONAL': {0: 1, 1: 2, 2: 3, 3: 5, 4: 6, 5: 7, 6: 9, 7: 10, 8: 11},
    'TETRAGONAL': {0: 1, 1: 2, 2: 4, 3: 5, 4: 7, 5: 8, 6: 10, 7: 11, 8: 13},
    'LOW': {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 11, 6: 13, 7: 15, 8: 17},
}

EXPECTED_HALF_INTEGER_COUNTS = {
    'CUBIC': {0.5: 1, 1.5: 1, 2.5: 2, 3.5: 3, 4.5: 3, 5.5: 4, 6.5: 5, 7.5: 5, 8.5: 6},
    'OTHERS': {0.5: 1, 1.5: 2, 2.5: 3, 3.5: 4, 4.5: 5, 5.5: 6, 6.5: 7, 7.5: 8, 8.5: 9},
}

SYMMETRY_FAMILIES = {
    'CUBIC': {'T', 'TD', 'O', 'OH', 'TH'},
    'HEXAGONAL': {'C3H', 'D3H', 'C6', 'C6H', 'C6V', 'D6', 'D6H'},
    'TRIGONAL': {'C3', 'S6', 'C3V', 'D3', 'D3D'},
    'TETRAGONAL': {'C4', 'S4', 'C4H', 'C4V', 'D4', 'D2D', 'D4H'},
    'LOW': {'C1', 'C1H', 'C2', 'C2H', 'C2V', 'D2', 'D2H', 'S2'},
}

def test_all_symmetries_covered():
    """Ensure all 32 allowed symmetries are accounted for."""
    all_grouped = set()
    for fam, syms in SYMMETRY_FAMILIES.items():
        all_grouped |= syms
    assert all_grouped == set(allowed_symmetries), f"Mismatch in symmetry groups: {all_grouped ^ set(allowed_symmetries)}"

def test_integer_j_counts():
    """Test integer J counts for all 32 point groups."""
    for family, sym_set in SYMMETRY_FAMILIES.items():
        expected_table = EXPECTED_INTEGER_COUNTS[family]
        for sym in sym_set:
            for J, expected in expected_table.items():
                actual = get_stark_levels_count(sym, J)
                assert actual == expected, (
                    f"Failed for symmetry '{sym}' ({family}) with J={J}: "
                    f"expected {expected}, got {actual}"
                )

def test_half_integer_j_counts():
    """Test half-integer J counts for all 32 point groups."""
    for family, sym_set in SYMMETRY_FAMILIES.items():
        expected_table = EXPECTED_HALF_INTEGER_COUNTS['CUBIC' if family == 'CUBIC' else 'OTHERS']
        for sym in sym_set:
            for J, expected in expected_table.items():
                actual = get_stark_levels_count(sym, J)
                assert actual == expected, (
                    f"Failed for symmetry '{sym}' ({family}) with J={J}: "
                    f"expected {expected}, got {actual}"
                )

def test_key_corrections():
    """Explicitly assert the specific regression points."""
    # 1. Trigonal integer J generic counts must NOT equal tetragonal counts for J in {2, 4, 5, 6, 7, 8}
    for trig_sym in ['C3', 'S6', 'C3V', 'D3', 'D3D']:
        assert get_stark_levels_count(trig_sym, 2) == 3, f"{trig_sym} J=2 must be 3"
        assert get_stark_levels_count(trig_sym, 4) == 6, f"{trig_sym} J=4 must be 6"
        assert get_stark_levels_count(trig_sym, 5) == 7, f"{trig_sym} J=5 must be 7"
        assert get_stark_levels_count(trig_sym, 6) == 9, f"{trig_sym} J=6 must be 9"
        assert get_stark_levels_count(trig_sym, 7) == 10, f"{trig_sym} J=7 must be 10"
        assert get_stark_levels_count(trig_sym, 8) == 11, f"{trig_sym} J=8 must be 11"

    # 2. Tetragonal integer J counts remain distinct
    for tet_sym in ['C4', 'S4', 'C4H', 'C4V', 'D4', 'D2D', 'D4H']:
        assert get_stark_levels_count(tet_sym, 2) == 4, f"{tet_sym} J=2 must be 4"
        assert get_stark_levels_count(tet_sym, 4) == 7, f"{tet_sym} J=4 must be 7"
        assert get_stark_levels_count(tet_sym, 5) == 8, f"{tet_sym} J=5 must be 8"
        assert get_stark_levels_count(tet_sym, 6) == 10, f"{tet_sym} J=6 must be 10"
        assert get_stark_levels_count(tet_sym, 7) == 11, f"{tet_sym} J=7 must be 11"
        assert get_stark_levels_count(tet_sym, 8) == 13, f"{tet_sym} J=8 must be 13"

    # 3. Cubic J=15/2 (7.5) must be 5 (two doublets and three quartets = 16 states)
    for cubic_sym in ['T', 'TD', 'O', 'OH', 'TH']:
        assert get_stark_levels_count(cubic_sym, 7.5) == 5, f"{cubic_sym} J=7.5 must be 5"

if __name__ == '__main__':
    test_all_symmetries_covered()
    test_integer_j_counts()
    test_half_integer_j_counts()
    test_key_corrections()
    print("ALL STARK LEVEL COUNTING REGRESSION TESTS PASSED SUCCESSFULLY (32 symmetries x 18 J values = 576 cases)!")
