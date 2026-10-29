"""
Dual Annealing Multi-Parameter Global Optimization Benchmark for YouZhiSan.
Author: Qi Dawei
Description:
    Evaluates 15 synergistic parameter configurations of SciPy's Generalized
    Simulated Annealing (dual_annealing) across 19 crystal field Hamiltonians
    (9 experimental datasets, 10 physically validated synthetic crystal systems).
    Benchmarking is executed across 76 problem instances: 19 stochastic initial
    guesses (fixed random seed) and 57 pre-optimized starting points obtained from
    the Basin-Hopping (10%_reflect) stage. Total benchmark volume: 76 * 15 = 1140 runs.
"""

import os
import sys
import copy
import time
import json
import random
import re
from multiprocessing import Pool, cpu_count
import numpy as np
from scipy.optimize import dual_annealing
from scipy import stats

# Setup environment paths relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
test_root = os.path.abspath(os.path.join(current_dir, ".."))
repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

site_pkg = r"C:\ProgramData\anaconda3\envs\PyCrystalField\Lib\site-packages"
if os.path.exists(site_pkg) and site_pkg not in sys.path:
    sys.path.insert(0, site_pkg)

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

if test_root not in sys.path:
    sys.path.insert(0, test_root)

import YZS_lib
from lbfgsb_benchmark.benchmark_lbfgsb import parse_real_case


# ==============================================================================
# 1. 57 Initial Guesses from Basin-Hopping (10%_reflect) Optimization Stage
# ==============================================================================
# 57 Initial Guesses from Basin-Hopping (10%_reflect) fitted CFP parameters
BASINHOPPING_FITTED_INITIAL_GUESSES = [   {   'case_name': '(Gruber J B et al, 1960) / #Gruber J B, Conway J G, 1960. Crys.txt',
        'dim': 4,
        'initial_cfp': {'AR20': 15.082093, 'AR40': -25.150115, 'AR60': 0.904535, 'AR66': 51.677866},
        'ion': 'TM3+',
        'symmetry': 'C6V',
        'trial_idx': 1},
    {   'case_name': '(Gruber J B et al, 1960) / #Gruber J B, Conway J G, 1960. Crys.txt',
        'dim': 4,
        'initial_cfp': {'AR20': 14.223019, 'AR40': 27.885473, 'AR60': -16.118672, 'AR66': 13.29734},
        'ion': 'TM3+',
        'symmetry': 'C6V',
        'trial_idx': 2},
    {   'case_name': '(Gruber J B et al, 1960) / #Gruber J B, Conway J G, 1960. Crys.txt',
        'dim': 4,
        'initial_cfp': {'AR20': 15.082081, 'AR40': -25.150119, 'AR60': 0.904537, 'AR66': 51.677891},
        'ion': 'TM3+',
        'symmetry': 'C6V',
        'trial_idx': 3},
    {   'case_name': '(Wong E Y et al, 1961) / (Wong E Y et al, 1961).txt',
        'dim': 4,
        'initial_cfp': {'AR20': 15.957156, 'AR40': -9.423646, 'AR60': -3.998947, 'AR66': -50.619858},
        'ion': 'TM3+',
        'symmetry': 'D3H',
        'trial_idx': 1},
    {   'case_name': '(Wong E Y et al, 1961) / (Wong E Y et al, 1961).txt',
        'dim': 4,
        'initial_cfp': {'AR20': 15.95715, 'AR40': -9.423647, 'AR60': -3.998947, 'AR66': -50.619853},
        'ion': 'TM3+',
        'symmetry': 'D3H',
        'trial_idx': 2},
    {   'case_name': '(Wong E Y et al, 1961) / (Wong E Y et al, 1961).txt',
        'dim': 4,
        'initial_cfp': {'AR20': 15.957157, 'AR40': -9.423647, 'AR60': -3.998948, 'AR66': 50.619858},
        'ion': 'TM3+',
        'symmetry': 'D3H',
        'trial_idx': 3},
    {   'case_name': '(Crosswhite H M et al, 1961) / (Crosswhite H M et al, 1961).txt',
        'dim': 4,
        'initial_cfp': {'AR20': 11.021147, 'AR40': -4.995898, 'AR60': -2.825919, 'AR66': 30.470431},
        'ion': 'DY3+',
        'symmetry': 'C6V',
        'trial_idx': 1},
    {   'case_name': '(Crosswhite H M et al, 1961) / (Crosswhite H M et al, 1961).txt',
        'dim': 4,
        'initial_cfp': {'AR20': 11.467015, 'AR40': -5.315456, 'AR60': -2.767066, 'AR66': -26.692984},
        'ion': 'DY3+',
        'symmetry': 'C6V',
        'trial_idx': 2},
    {   'case_name': '(Crosswhite H M et al, 1961) / (Crosswhite H M et al, 1961).txt',
        'dim': 4,
        'initial_cfp': {'AR20': 11.021147, 'AR40': -4.995899, 'AR60': -2.825919, 'AR66': 30.47043},
        'ion': 'DY3+',
        'symmetry': 'C6V',
        'trial_idx': 3},
    {   'case_name': '(Erath E H, 1961) / (Erath E H, 1961).txt',
        'dim': 4,
        'initial_cfp': {'AR20': 15.914136, 'AR40': -9.826589, 'AR60': -3.771022, 'AR66': 48.505897},
        'ion': 'ER3+',
        'symmetry': 'D3H',
        'trial_idx': 1},
    {   'case_name': '(Erath E H, 1961) / (Erath E H, 1961).txt',
        'dim': 4,
        'initial_cfp': {'AR20': 15.914136, 'AR40': -9.826589, 'AR60': -3.771022, 'AR66': 48.505897},
        'ion': 'ER3+',
        'symmetry': 'D3H',
        'trial_idx': 2},
    {   'case_name': '(Erath E H, 1961) / (Erath E H, 1961).txt',
        'dim': 4,
        'initial_cfp': {'AR20': 15.914125, 'AR40': -9.826583, 'AR60': -3.771018, 'AR66': 48.505913},
        'ion': 'ER3+',
        'symmetry': 'D3H',
        'trial_idx': 3},
    {   'case_name': '(Judd B R et al, 1957) / (Judd B R et al, 1957).txt',
        'dim': 4,
        'initial_cfp': {'AR20': -5.299577, 'AR40': -7.98871, 'AR60': 4.993983, 'AR66': 14.626129},
        'ion': 'PR3+',
        'symmetry': 'D6',
        'trial_idx': 1},
    {   'case_name': '(Judd B R et al, 1957) / (Judd B R et al, 1957).txt',
        'dim': 4,
        'initial_cfp': {'AR20': -5.267847, 'AR40': -7.688286, 'AR60': 5.332539, 'AR66': 16.2588},
        'ion': 'PR3+',
        'symmetry': 'D6',
        'trial_idx': 2},
    {   'case_name': '(Judd B R et al, 1957) / (Judd B R et al, 1957).txt',
        'dim': 4,
        'initial_cfp': {'AR20': -5.299573, 'AR40': -7.98871, 'AR60': 4.993985, 'AR66': -14.626114},
        'ion': 'PR3+',
        'symmetry': 'D6',
        'trial_idx': 3},
    {   'case_name': '(Koningstein J A et al, 1964) / (Koningstein J A et al, 1964).txt',
        'dim': 5,
        'initial_cfp': {   'AR20': -22.607845,
                           'AR40': -32.167458,
                           'AR44': -8.155505,
                           'AR60': 0.064383,
                           'AR64': -108.129991},
        'ion': 'ND3+',
        'symmetry': 'C4V',
        'trial_idx': 1},
    {   'case_name': '(Koningstein J A et al, 1964) / (Koningstein J A et al, 1964).txt',
        'dim': 5,
        'initial_cfp': {'AR20': 39.80472, 'AR40': 15.109771, 'AR44': 81.778913, 'AR60': -18.879884, 'AR64': 131.234189},
        'ion': 'ND3+',
        'symmetry': 'C4V',
        'trial_idx': 2},
    {   'case_name': '(Koningstein J A et al, 1964) / (Koningstein J A et al, 1964).txt',
        'dim': 5,
        'initial_cfp': {   'AR20': 37.971088,
                           'AR40': 13.167694,
                           'AR44': -124.091803,
                           'AR60': -18.594732,
                           'AR64': -35.556485},
        'ion': 'ND3+',
        'symmetry': 'C4V',
        'trial_idx': 3},
    {   'case_name': '(Capobianco J A et al, 1997) / (Capobianco J A et al, 1997).txt',
        'dim': 5,
        'initial_cfp': {   'AR20': -16.043355,
                           'AR40': 1.467919,
                           'AR44': -124.153047,
                           'AR60': -2.668063,
                           'AR64': -24.320829},
        'ion': 'ER3+',
        'symmetry': 'D2D',
        'trial_idx': 1},
    {   'case_name': '(Capobianco J A et al, 1997) / (Capobianco J A et al, 1997).txt',
        'dim': 5,
        'initial_cfp': {   'AR20': -16.043394,
                           'AR40': 1.467918,
                           'AR44': -124.153049,
                           'AR60': -2.668061,
                           'AR64': -24.320824},
        'ion': 'ER3+',
        'symmetry': 'D2D',
        'trial_idx': 2},
    {   'case_name': '(Capobianco J A et al, 1997) / (Capobianco J A et al, 1997).txt',
        'dim': 5,
        'initial_cfp': {   'AR20': -16.043375,
                           'AR40': 1.467896,
                           'AR44': -124.153131,
                           'AR60': -2.668057,
                           'AR64': -24.320809},
        'ion': 'ER3+',
        'symmetry': 'D2D',
        'trial_idx': 3},
    {   'case_name': '(Popova M N et al, 2007) / (Popova M N et al, 2007).txt',
        'dim': 6,
        'initial_cfp': {   'AR20': 31.515224,
                           'AR40': -16.912897,
                           'AR43': 222.886296,
                           'AR60': -3.272372,
                           'AR63': 83.068212,
                           'AR66': -8.403949},
        'ion': 'ND3+',
        'symmetry': 'D3',
        'trial_idx': 1},
    {   'case_name': '(Popova M N et al, 2007) / (Popova M N et al, 2007).txt',
        'dim': 6,
        'initial_cfp': {   'AR20': 31.515184,
                           'AR40': -16.912907,
                           'AR43': -222.886772,
                           'AR60': -3.272371,
                           'AR63': -83.068119,
                           'AR66': -8.403859},
        'ion': 'ND3+',
        'symmetry': 'D3',
        'trial_idx': 2},
    {   'case_name': '(Popova M N et al, 2007) / (Popova M N et al, 2007).txt',
        'dim': 6,
        'initial_cfp': {   'AR20': -41.027499,
                           'AR40': -11.102061,
                           'AR43': 135.267639,
                           'AR60': -6.852562,
                           'AR63': 40.745339,
                           'AR66': 23.861354},
        'ion': 'ND3+',
        'symmetry': 'D3',
        'trial_idx': 3},
    {   'case_name': '(Chen X Y et al, 2005) / (Chen X Y et al, 2005).txt',
        'dim': 9,
        'initial_cfp': {   'AR20': -19.462843,
                           'AR22': -26.181654,
                           'AR40': -28.357181,
                           'AR42': -100.0,
                           'AR44': 100.0,
                           'AR60': 3.212317,
                           'AR62': 100.0,
                           'AR64': -8.064938,
                           'AR66': 65.840029},
        'ion': 'EU3+',
        'symmetry': 'C2V',
        'trial_idx': 1},
    {   'case_name': '(Chen X Y et al, 2005) / (Chen X Y et al, 2005).txt',
        'dim': 9,
        'initial_cfp': {   'AR20': -12.451554,
                           'AR22': -54.821442,
                           'AR40': -39.795664,
                           'AR42': 12.531095,
                           'AR44': 100.0,
                           'AR60': 2.379271,
                           'AR62': -3.162611,
                           'AR64': 88.200096,
                           'AR66': -10.146688},
        'ion': 'EU3+',
        'symmetry': 'C2V',
        'trial_idx': 2},
    {   'case_name': '(Chen X Y et al, 2005) / (Chen X Y et al, 2005).txt',
        'dim': 9,
        'initial_cfp': {   'AR20': -20.488223,
                           'AR22': 40.894435,
                           'AR40': -39.700646,
                           'AR42': -18.759811,
                           'AR44': -100.0,
                           'AR60': 2.499322,
                           'AR62': 10.234938,
                           'AR64': -87.545262,
                           'AR66': 10.574805},
        'ion': 'EU3+',
        'symmetry': 'C2V',
        'trial_idx': 3},
    {   'case_name': 'Synthetic_01 [Dy3+ in C6V, 4 params, 6H5/2,6F7/2]',
        'dim': 4,
        'initial_cfp': {'AR20': -67.357048, 'AR40': -13.918122, 'AR60': 38.130838, 'AR66': -32.888078},
        'ion': 'Dy3+',
        'symmetry': 'C6V',
        'trial_idx': 1},
    {   'case_name': 'Synthetic_01 [Dy3+ in C6V, 4 params, 6H5/2,6F7/2]',
        'dim': 4,
        'initial_cfp': {'AR20': -67.357048, 'AR40': -13.918122, 'AR60': 38.130838, 'AR66': -32.888081},
        'ion': 'Dy3+',
        'symmetry': 'C6V',
        'trial_idx': 2},
    {   'case_name': 'Synthetic_01 [Dy3+ in C6V, 4 params, 6H5/2,6F7/2]',
        'dim': 4,
        'initial_cfp': {'AR20': -67.357048, 'AR40': -13.918122, 'AR60': 38.130838, 'AR66': 32.888083},
        'ion': 'Dy3+',
        'symmetry': 'C6V',
        'trial_idx': 3},
    {   'case_name': 'Synthetic_02 [Sm3+ in C6V, 4 params, 6H7/2,6H5/2]',
        'dim': 4,
        'initial_cfp': {'AR20': 84.198591, 'AR40': -29.196639, 'AR60': 9.650083, 'AR66': 100.0},
        'ion': 'Sm3+',
        'symmetry': 'C6V',
        'trial_idx': 1},
    {   'case_name': 'Synthetic_02 [Sm3+ in C6V, 4 params, 6H7/2,6H5/2]',
        'dim': 4,
        'initial_cfp': {'AR20': -68.173372, 'AR40': -42.971026, 'AR60': -21.527247, 'AR66': -0.001124},
        'ion': 'Sm3+',
        'symmetry': 'C6V',
        'trial_idx': 2},
    {   'case_name': 'Synthetic_02 [Sm3+ in C6V, 4 params, 6H7/2,6H5/2]',
        'dim': 4,
        'initial_cfp': {'AR20': -68.173376, 'AR40': -42.971025, 'AR60': -21.527247, 'AR66': -0.001476},
        'ion': 'Sm3+',
        'symmetry': 'C6V',
        'trial_idx': 3},
    {   'case_name': 'Synthetic_03 [Pm3+ in D4, 5 params, 5I4]',
        'dim': 5,
        'initial_cfp': {'AR20': 94.581236, 'AR40': -72.291986, 'AR44': 8.789161, 'AR60': 4.951511, 'AR64': 10.902829},
        'ion': 'Pm3+',
        'symmetry': 'D4',
        'trial_idx': 1},
    {   'case_name': 'Synthetic_03 [Pm3+ in D4, 5 params, 5I4]',
        'dim': 5,
        'initial_cfp': {'AR20': -106.21407, 'AR40': -29.114058, 'AR44': -150.0, 'AR60': 0.266881, 'AR64': -42.064157},
        'ion': 'Pm3+',
        'symmetry': 'D4',
        'trial_idx': 2},
    {   'case_name': 'Synthetic_03 [Pm3+ in D4, 5 params, 5I4]',
        'dim': 5,
        'initial_cfp': {'AR20': -106.21407, 'AR40': -29.114062, 'AR44': -150.0, 'AR60': 0.266881, 'AR64': -42.064157},
        'ion': 'Pm3+',
        'symmetry': 'D4',
        'trial_idx': 3},
    {   'case_name': 'Synthetic_04 [Ce3+ in D4, 5 params, 2F5/2]',
        'dim': 5,
        'initial_cfp': {'AR20': -90.818978, 'AR40': 5.274984, 'AR44': 15.979803, 'AR60': 3.112274, 'AR64': 61.768123},
        'ion': 'Ce3+',
        'symmetry': 'D4',
        'trial_idx': 1},
    {   'case_name': 'Synthetic_04 [Ce3+ in D4, 5 params, 2F5/2]',
        'dim': 5,
        'initial_cfp': {   'AR20': -75.501675,
                           'AR40': 23.627542,
                           'AR44': -75.926195,
                           'AR60': 92.201027,
                           'AR64': 47.412843},
        'ion': 'Ce3+',
        'symmetry': 'D4',
        'trial_idx': 2},
    {   'case_name': 'Synthetic_04 [Ce3+ in D4, 5 params, 2F5/2]',
        'dim': 5,
        'initial_cfp': {   'AR20': -12.533645,
                           'AR40': -41.696216,
                           'AR44': 129.765318,
                           'AR60': 19.682369,
                           'AR64': 3.696489},
        'ion': 'Ce3+',
        'symmetry': 'D4',
        'trial_idx': 3},
    {   'case_name': 'Synthetic_05 [Dy3+ in D2D, 5 params, 6H7/2,6H13/2]',
        'dim': 5,
        'initial_cfp': {'AR20': -7.091725, 'AR40': 29.212401, 'AR44': -60.701293, 'AR60': 14.63363, 'AR64': -1.622913},
        'ion': 'Dy3+',
        'symmetry': 'D2D',
        'trial_idx': 1},
    {   'case_name': 'Synthetic_05 [Dy3+ in D2D, 5 params, 6H7/2,6H13/2]',
        'dim': 5,
        'initial_cfp': {   'AR20': -13.374457,
                           'AR40': -36.465855,
                           'AR44': -87.344901,
                           'AR60': -8.231038,
                           'AR64': 92.92825},
        'ion': 'Dy3+',
        'symmetry': 'D2D',
        'trial_idx': 2},
    {   'case_name': 'Synthetic_05 [Dy3+ in D2D, 5 params, 6H7/2,6H13/2]',
        'dim': 5,
        'initial_cfp': {'AR20': -7.091725, 'AR40': 29.212399, 'AR44': -60.70129, 'AR60': 14.63363, 'AR64': -1.622922},
        'ion': 'Dy3+',
        'symmetry': 'D2D',
        'trial_idx': 3},
    {   'case_name': 'Synthetic_06 [Pr3+ in D2D, 5 params, 3P1,1G4]',
        'dim': 5,
        'initial_cfp': {'AR20': -7.873395, 'AR40': -39.449651, 'AR44': -17.754926, 'AR60': 5.14982, 'AR64': -6.326888},
        'ion': 'Pr3+',
        'symmetry': 'D2D',
        'trial_idx': 1},
    {   'case_name': 'Synthetic_06 [Pr3+ in D2D, 5 params, 3P1,1G4]',
        'dim': 5,
        'initial_cfp': {'AR20': -8.851713, 'AR40': -10.914229, 'AR44': 120.043687, 'AR60': -7.860001, 'AR64': -100.0},
        'ion': 'Pr3+',
        'symmetry': 'D2D',
        'trial_idx': 2},
    {   'case_name': 'Synthetic_06 [Pr3+ in D2D, 5 params, 3P1,1G4]',
        'dim': 5,
        'initial_cfp': {'AR20': -7.918694, 'AR40': -38.815654, 'AR44': -1.775125, 'AR60': 5.957688, 'AR64': 0.399878},
        'ion': 'Pr3+',
        'symmetry': 'D2D',
        'trial_idx': 3},
    {   'case_name': 'Synthetic_07 [Pr3+ in D3, 6 params, 1G4,1D2]',
        'dim': 6,
        'initial_cfp': {   'AR20': 36.505837,
                           'AR40': 25.61826,
                           'AR43': -126.962148,
                           'AR60': -2.661829,
                           'AR63': 72.833401,
                           'AR66': 52.631588},
        'ion': 'Pr3+',
        'symmetry': 'D3',
        'trial_idx': 1},
    {   'case_name': 'Synthetic_07 [Pr3+ in D3, 6 params, 1G4,1D2]',
        'dim': 6,
        'initial_cfp': {   'AR20': -68.749749,
                           'AR40': -4.798779,
                           'AR43': -129.052346,
                           'AR60': -8.19176,
                           'AR63': 100.0,
                           'AR66': 100.0},
        'ion': 'Pr3+',
        'symmetry': 'D3',
        'trial_idx': 2},
    {   'case_name': 'Synthetic_07 [Pr3+ in D3, 6 params, 1G4,1D2]',
        'dim': 6,
        'initial_cfp': {   'AR20': 38.109033,
                           'AR40': 26.161069,
                           'AR43': 66.719777,
                           'AR60': 2.604016,
                           'AR63': 90.52873,
                           'AR66': -43.626682},
        'ion': 'Pr3+',
        'symmetry': 'D3',
        'trial_idx': 3},
    {   'case_name': 'Synthetic_08 [Eu3+ in D3, 6 params, 7F4,7F3]',
        'dim': 6,
        'initial_cfp': {   'AR20': -69.340236,
                           'AR40': -47.571789,
                           'AR43': 28.735453,
                           'AR60': 19.377364,
                           'AR63': 29.461646,
                           'AR66': 33.406316},
        'ion': 'Eu3+',
        'symmetry': 'D3',
        'trial_idx': 1},
    {   'case_name': 'Synthetic_08 [Eu3+ in D3, 6 params, 7F4,7F3]',
        'dim': 6,
        'initial_cfp': {   'AR20': -69.340232,
                           'AR40': -47.571787,
                           'AR43': 28.734861,
                           'AR60': 19.377364,
                           'AR63': 29.461933,
                           'AR66': 33.406322},
        'ion': 'Eu3+',
        'symmetry': 'D3',
        'trial_idx': 2},
    {   'case_name': 'Synthetic_08 [Eu3+ in D3, 6 params, 7F4,7F3]',
        'dim': 6,
        'initial_cfp': {   'AR20': -47.364231,
                           'AR40': 36.031529,
                           'AR43': -136.314452,
                           'AR60': -19.996786,
                           'AR63': -100.0,
                           'AR66': -98.884646},
        'ion': 'Eu3+',
        'symmetry': 'D3',
        'trial_idx': 3},
    {   'case_name': 'Synthetic_09 [Nd3+ in C2V, 9 params, 4G5/2,4G7/2]',
        'dim': 9,
        'initial_cfp': {   'AR20': -167.716309,
                           'AR22': 94.765823,
                           'AR40': -77.374418,
                           'AR42': 48.148923,
                           'AR44': -114.405395,
                           'AR60': -1.710982,
                           'AR62': 90.048818,
                           'AR64': -28.96351,
                           'AR66': 23.576903},
        'ion': 'Nd3+',
        'symmetry': 'C2V',
        'trial_idx': 1},
    {   'case_name': 'Synthetic_09 [Nd3+ in C2V, 9 params, 4G5/2,4G7/2]',
        'dim': 9,
        'initial_cfp': {   'AR20': -58.919533,
                           'AR22': 53.992835,
                           'AR40': -76.687893,
                           'AR42': 36.976185,
                           'AR44': -137.883942,
                           'AR60': 12.628777,
                           'AR62': -15.308854,
                           'AR64': -12.405892,
                           'AR66': -38.931106},
        'ion': 'Nd3+',
        'symmetry': 'C2V',
        'trial_idx': 2},
    {   'case_name': 'Synthetic_09 [Nd3+ in C2V, 9 params, 4G5/2,4G7/2]',
        'dim': 9,
        'initial_cfp': {   'AR20': -200.0,
                           'AR22': 200.0,
                           'AR40': 47.184356,
                           'AR42': 150.0,
                           'AR44': -116.936259,
                           'AR60': 8.814328,
                           'AR62': 35.649226,
                           'AR64': 89.438354,
                           'AR66': -26.662568},
        'ion': 'Nd3+',
        'symmetry': 'C2V',
        'trial_idx': 3},
    {   'case_name': 'Synthetic_10 [Yb3+ in C2V, 9 params, 2F7/2,2F5/2]',
        'dim': 9,
        'initial_cfp': {   'AR20': -20.834542,
                           'AR22': 29.882099,
                           'AR40': -18.763337,
                           'AR42': -10.195305,
                           'AR44': -115.533432,
                           'AR60': 37.491336,
                           'AR62': -46.507229,
                           'AR64': 44.948557,
                           'AR66': -98.021769},
        'ion': 'Yb3+',
        'symmetry': 'C2V',
        'trial_idx': 1},
    {   'case_name': 'Synthetic_10 [Yb3+ in C2V, 9 params, 2F7/2,2F5/2]',
        'dim': 9,
        'initial_cfp': {   'AR20': 19.620703,
                           'AR22': 58.962059,
                           'AR40': 11.16441,
                           'AR42': -77.476704,
                           'AR44': -57.538467,
                           'AR60': -36.59144,
                           'AR62': -73.201614,
                           'AR64': 65.159848,
                           'AR66': -94.721338},
        'ion': 'Yb3+',
        'symmetry': 'C2V',
        'trial_idx': 2},
    {   'case_name': 'Synthetic_10 [Yb3+ in C2V, 9 params, 2F7/2,2F5/2]',
        'dim': 9,
        'initial_cfp': {   'AR20': 35.029094,
                           'AR22': -11.77324,
                           'AR40': 18.402821,
                           'AR42': -54.71824,
                           'AR44': 60.347879,
                           'AR60': 37.492376,
                           'AR62': 67.559606,
                           'AR64': -21.335428,
                           'AR66': 95.055171},
        'ion': 'Yb3+',
        'symmetry': 'C2V',
        'trial_idx': 3}]

# ==============================================================================
# 2. Phase 2 Dual Annealing Configurations (15 Synergy Parameter Sets)
# ==============================================================================
PHASE2_CONFIGS = [
    {
        "name": "C01_Baseline_Default",
        "maxiter": 1000,
        "visit": 2.62,
        "restart_temp_ratio": 2e-5,
        "initial_temp": 5230.0,
        "accept": -5.0,
        "description": "Standard SciPy GSA baseline parameters"
    },
    {
        "name": "C02_Restart_1e-4",
        "maxiter": 1000,
        "visit": 2.62,
        "restart_temp_ratio": 1e-4,
        "initial_temp": 5230.0,
        "accept": -5.0,
        "description": "Moderate periodic re-annealing threshold"
    },
    {
        "name": "C03_Restart_5e-4",
        "maxiter": 1000,
        "visit": 2.62,
        "restart_temp_ratio": 5e-4,
        "initial_temp": 5230.0,
        "accept": -5.0,
        "description": "Intermediate cyclic re-annealing threshold"
    },
    {
        "name": "C04_Restart_1e-3_Standard",
        "maxiter": 1000,
        "visit": 2.62,
        "restart_temp_ratio": 1e-3,
        "initial_temp": 5230.0,
        "accept": -5.0,
        "description": "Active cyclic re-annealing at canonical Tsallis parameter"
    },
    {
        "name": "C05_Restart_2e-3_Frequent",
        "maxiter": 1000,
        "visit": 2.62,
        "restart_temp_ratio": 2e-3,
        "initial_temp": 5230.0,
        "accept": -5.0,
        "description": "Frequent cyclic re-annealing with rapid temperature reset"
    },
    {
        "name": "C06_Gentle_Visit_2.50_Restart_1e-3",
        "maxiter": 1000,
        "visit": 2.50,
        "restart_temp_ratio": 1e-3,
        "initial_temp": 5230.0,
        "accept": -5.0,
        "description": "Sub-Gaussian short-range exploration with cyclic restart"
    },
    {
        "name": "C07_Moderate_Visit_2.70_Restart_1e-3",
        "maxiter": 1000,
        "visit": 2.70,
        "restart_temp_ratio": 1e-3,
        "initial_temp": 5230.0,
        "accept": -5.0,
        "description": "Mildly broadened Tsallis distribution with cyclic restart"
    },
    {
        "name": "C08_Extended_Visit_2.75_Restart_1e-3",
        "maxiter": 1000,
        "visit": 2.75,
        "restart_temp_ratio": 1e-3,
        "initial_temp": 5230.0,
        "accept": -5.0,
        "description": "Extended jump exploration with cyclic restart"
    },
    {
        "name": "C09_Levy_Visit_2.90_Restart_1e-3",
        "maxiter": 1000,
        "visit": 2.90,
        "restart_temp_ratio": 1e-3,
        "initial_temp": 5230.0,
        "accept": -5.0,
        "description": "Heavy-tailed Levy flight regime with cyclic restart"
    },
    {
        "name": "C10_Warm_Temp_10000_Restart_1e-3",
        "maxiter": 1000,
        "visit": 2.62,
        "restart_temp_ratio": 1e-3,
        "initial_temp": 10000.0,
        "accept": -5.0,
        "description": "Elevated initial kinetic temperature (T0=10000) with cyclic restart"
    },
    {
        "name": "C11_Hot_Temp_20000_Restart_1e-3",
        "maxiter": 1000,
        "visit": 2.62,
        "restart_temp_ratio": 1e-3,
        "initial_temp": 20000.0,
        "accept": -5.0,
        "description": "High kinetic exploration temperature (T0=20000) with cyclic restart"
    },
    {
        "name": "C12_Selective_Accept_m15_Restart_1e-3",
        "maxiter": 1000,
        "visit": 2.62,
        "restart_temp_ratio": 1e-3,
        "initial_temp": 5230.0,
        "accept": -15.0,
        "description": "Stricter uphill transition acceptance (qa=-15) with cyclic restart"
    },
    {
        "name": "C13_Strict_Accept_m30_Restart_1e-3",
        "maxiter": 1000,
        "visit": 2.62,
        "restart_temp_ratio": 1e-3,
        "initial_temp": 5230.0,
        "accept": -30.0,
        "description": "Steep downhill-biased acceptance (qa=-30) with cyclic restart"
    },
    {
        "name": "C14_Fast_Budget_500_Restart_1e-3",
        "maxiter": 500,
        "visit": 2.62,
        "restart_temp_ratio": 1e-3,
        "initial_temp": 5230.0,
        "accept": -5.0,
        "description": "Truncated budget (maxiter=500) for rapid screening"
    },
    {
        "name": "C15_Deep_Budget_1500_Restart_1e-3",
        "maxiter": 1500,
        "visit": 2.62,
        "restart_temp_ratio": 1e-3,
        "initial_temp": 5230.0,
        "accept": -5.0,
        "description": "Extended budget (maxiter=1500) for thorough global exploration"
    },
]


# ==============================================================================
# 3. Dataset Specification (19 Crystal Field Systems)
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
# 4. Worker Function
# ==============================================================================
def run_da_task(task_args):
    """
    Executes a single dual_annealing run for one problem instance and configuration.
    """
    sys.stdout = open(os.devnull, 'w')
    instance_meta, cfg = task_args

    instance_idx = instance_meta["instance_idx"]
    case_info = instance_meta["case_info"]
    x0 = instance_meta["x0"]
    instance_seed = instance_meta["instance_seed"]
    origin = instance_meta["origin"]
    trial_idx = instance_meta["trial_idx"]

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

    t0 = time.time()
    try:
        f0 = float(obj(x0))
    except Exception:
        f0 = 1e12

    try:
        res = dual_annealing(
            func=obj,
            bounds=bounds,
            x0=x0.copy(),
            maxiter=cfg["maxiter"],
            visit=cfg["visit"],
            restart_temp_ratio=cfg["restart_temp_ratio"],
            initial_temp=cfg["initial_temp"],
            accept=cfg["accept"],
            seed=instance_seed,
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "jac": "3-point",
                "bounds": bounds,
                "options": {"finite_diff_rel_step": 1e-4},
            }
        )
        elapsed = time.time() - t0
        final_ssd = float(res.fun) if np.isfinite(res.fun) else 1e12
        msg = res.message[0] if isinstance(res.message, list) and res.message else str(res.message)

        # Reconstruct full CFP dictionary
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

        return {
            "instance_idx": instance_idx,
            "case_name": case_info["case_name"],
            "symmetry": case_info["symmetry"],
            "dim": len(bounds),
            "origin": origin,
            "trial_idx": trial_idx,
            "cfg_name": cfg["name"],
            "f0": f0,
            "final_ssd": final_ssd,
            "ssd_drop": float(max(0.0, f0 - final_ssd)),
            "nfev": int(res.nfev),
            "nit": int(res.nit),
            "time_sec": float(elapsed),
            "success": bool(res.success),
            "message": msg,
            "best_x": res.x.tolist(),
            "independent_cfp": ind_cfp,
            "full_cfp": full_cfp
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "instance_idx": instance_idx,
            "case_name": case_info["case_name"],
            "symmetry": case_info["symmetry"],
            "dim": len(bounds),
            "origin": origin,
            "trial_idx": trial_idx,
            "cfg_name": cfg["name"],
            "f0": f0,
            "final_ssd": 1e12,
            "ssd_drop": 0.0,
            "nfev": 0,
            "nit": 0,
            "time_sec": float(elapsed),
            "success": False,
            "message": str(e),
            "best_x": [],
            "independent_cfp": {},
            "full_cfp": {}
        }


# ==============================================================================
# 5. Statistical Analysis Engine
# ==============================================================================
def compute_statistics_subset(results_dict, baseline_name, instance_indices):
    stats_out = {}
    cfg_names = list(results_dict.keys())
    base_runs = results_dict.get(baseline_name, {})

    for name in cfg_names:
        runs = results_dict[name]
        ssds = [runs[i]["final_ssd"] for i in instance_indices if i in runs]
        times = [runs[i]["time_sec"] for i in instance_indices if i in runs]
        nfevs = [runs[i]["nfev"] for i in instance_indices if i in runs]
        drops = [runs[i]["ssd_drop"] for i in instance_indices if i in runs]

        mean_ssd = float(np.mean(ssds)) if ssds else 0.0
        median_ssd = float(np.median(ssds)) if ssds else 0.0
        mean_time = float(np.mean(times)) if times else 0.0
        mean_nfev = float(np.mean(nfevs)) if nfevs else 0.0
        mean_drop = float(np.mean(drops)) if drops else 0.0

        # Paired comparison vs Baseline
        diffs = []
        wins, losses, ties = 0, 0, 0
        for i in instance_indices:
            if i in runs and i in base_runs:
                s_test = runs[i]["final_ssd"]
                s_base = base_runs[i]["final_ssd"]
                d = s_test - s_base
                diffs.append(d)
                if d < -1e-4:
                    wins += 1
                elif d > 1e-4:
                    losses += 1
                else:
                    ties += 1

        diffs = np.array(diffs)
        mean_diff = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
        std_err_diff = float(stats.sem(diffs)) if len(diffs) > 1 else 0.0

        p_ttest = 1.0
        p_wilcoxon = 1.0
        if len(diffs) > 1 and np.any(diffs != 0):
            try:
                t_stat, p_ttest = stats.ttest_rel(
                    [runs[i]["final_ssd"] for i in instance_indices if i in runs and i in base_runs],
                    [base_runs[i]["final_ssd"] for i in instance_indices if i in runs and i in base_runs]
                )
            except Exception:
                pass
            try:
                nonzero_diffs = diffs[diffs != 0]
                if len(nonzero_diffs) > 5:
                    w_stat, p_wilcoxon = stats.wilcoxon(nonzero_diffs)
            except Exception:
                pass

        stats_out[name] = {
            "count": len(ssds),
            "mean_ssd": mean_ssd,
            "median_ssd": median_ssd,
            "mean_time_sec": mean_time,
            "mean_nfev": mean_nfev,
            "mean_ssd_drop": mean_drop,
            "paired_diff_mean": mean_diff,
            "paired_diff_stderr": std_err_diff,
            "p_value_ttest": float(p_ttest),
            "p_value_wilcoxon": float(p_wilcoxon),
            "wins": wins,
            "losses": losses,
            "ties": ties
        }

    return stats_out


def compute_statistics(results_dict, baseline_name, all_instances):
    all_indices = list(range(len(all_instances)))
    random_indices = [i for i, inst in enumerate(all_instances) if inst["origin"] == "random_fixed_seed"]
    bh_indices = [i for i, inst in enumerate(all_instances) if inst["origin"] == "basinhopping_10pct_reflect"]

    return {
        "overall": compute_statistics_subset(results_dict, baseline_name, all_indices),
        "random_init_19": compute_statistics_subset(results_dict, baseline_name, random_indices),
        "bh_fitted_init_57": compute_statistics_subset(results_dict, baseline_name, bh_indices)
    }


# ==============================================================================
# 6. Execution Engine with Incremental JSONL Persistence
# ==============================================================================
def execute_benchmark(configs, instances, n_workers, jsonl_file):
    total_tasks = len(instances) * len(configs)
    print("\n" + "=" * 80)
    print(f"  Executing Phase 2 Benchmark: {len(configs)} Configurations across {len(instances)} Instances")
    print(f"  Breakdown: 19 Random Initial Guesses + 57 Basin-Hopping Refined Initial Guesses")
    print(f"  Total planned runs: ({len(instances)} x {len(configs)}) = {total_tasks}")
    print("=" * 80)

    # 1. Load already completed tasks from JSONL (incremental resume)
    completed = {}
    if os.path.exists(jsonl_file):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        key = (record["cfg_name"], record["instance_idx"])
                        completed[key] = record
                    except Exception:
                        pass
        print(f"  [Resume] Loaded {len(completed)} existing runs from {os.path.basename(jsonl_file)}")

    # 2. Build task queue, skipping completed
    tasks_to_run = []
    for cfg in configs:
        for inst in instances:
            key = (cfg["name"], inst["instance_idx"])
            if key not in completed:
                tasks_to_run.append((inst, cfg))

    remaining_tasks = len(tasks_to_run)
    print(f"  [Queue] {remaining_tasks} remaining runs to execute out of {total_tasks} total.")

    # 3. Open JSONL file for continuous appending
    f_out = open(jsonl_file, "a", encoding="utf-8")

    t_start = time.time()
    done_count = len(completed)

    if remaining_tasks > 0:
        with Pool(processes=n_workers) as pool:
            for res in pool.imap_unordered(run_da_task, tasks_to_run, chunksize=1):
                f_out.write(json.dumps(res) + "\n")
                f_out.flush()
                key = (res["cfg_name"], res["instance_idx"])
                completed[key] = res
                done_count += 1
                if done_count % 10 == 0 or done_count == total_tasks:
                    pct = (done_count / total_tasks) * 100.0
                    elapsed = time.time() - t_start
                    print(f"  [Progress] {done_count}/{total_tasks} ({pct:.1f}%) | Elapsed: {elapsed:.1f}s | Latest: {res['cfg_name'][:24]} on Inst {res['instance_idx']:02d} [{res['origin'][:6]}] (SSD={res['final_ssd']:.3f})")

    f_out.close()
    wall_time = time.time() - t_start
    print(f"  Phase 2 benchmark completed in {wall_time:.2f} s ({wall_time/60.0:.2f} min).\n")

    # Group into structured list and dictionary
    raw_list = list(completed.values())
    by_cfg = {cfg["name"]: {} for cfg in configs}
    for res in raw_list:
        if res["cfg_name"] in by_cfg:
            by_cfg[res["cfg_name"]][res["instance_idx"]] = res

    return raw_list, by_cfg, wall_time


# ==============================================================================
# 7. Main Entry Point
# ==============================================================================
def main():
    print("=" * 80)
    print("  YouZhiSan Dual Annealing Optimization Benchmark (Phase 2 Synergy Matrix)")
    print(f"  Available CPU Cores: {cpu_count()} | Worker Pool: 16 | Persistence: Incremental JSONL")
    print("=" * 80 + "\n")

    os.makedirs(current_dir, exist_ok=True)

    # 1. Load Problem Instances (19 Crystal Field Systems)
    real_cases = load_real_cases()
    syn_cases = generate_synthetic_cases(seed=2026)
    all_cases_list = real_cases + syn_cases
    cases_by_name = {c["case_name"]: c for c in all_cases_list}
    print(f"[1/4] Loaded {len(real_cases)} real experimental cases and {len(syn_cases)} synthetic cases.")
    print(f"      Total unique crystal field Hamiltonians: {len(all_cases_list)}")

    # 2. Construct 76 Problem Instances (19 Random + 57 Basin-Hopping Initial Guesses)
    instances = []

    # Part A: 19 stochastic initial guesses (fixed seed 42, 1 per Hamiltonian)
    random.seed(42)
    np.random.seed(42)
    for idx, case in enumerate(all_cases_list):
        bounds = case["bounds"]
        p0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds], dtype=float)
        instance_seed = 1000 + idx * 17
        instances.append({
            "instance_idx": len(instances),
            "case_info": case,
            "x0": p0,
            "instance_seed": instance_seed,
            "origin": "random_fixed_seed",
            "trial_idx": 1
        })

    # Part B: 57 Basin-Hopping (10%_reflect) pre-optimized initial guesses
    for entry in BASINHOPPING_FITTED_INITIAL_GUESSES:
        cname = entry["case_name"]
        case = cases_by_name[cname]
        bounds = case["bounds"]
        constraints = case["constraints"]
        all_keys = list(case["Ar_dict"].keys())
        total_vars = len(all_keys)
        ind_indices = sorted(list(set(range(total_vars)) - set(constraints.keys())))
        ind_keys = [all_keys[i] for i in ind_indices]

        p0 = np.array([entry["initial_cfp"][k] for k in ind_keys], dtype=float)
        for i, (low, high) in enumerate(bounds):
            p0[i] = np.clip(p0[i], low, high)

        idx = len(instances)
        instance_seed = 1000 + idx * 17
        instances.append({
            "instance_idx": idx,
            "case_info": case,
            "x0": p0,
            "instance_seed": instance_seed,
            "origin": "basinhopping_10pct_reflect",
            "trial_idx": entry["trial_idx"]
        })

    print(f"[2/4] Constructed {len(instances)} problem instances: 19 random + 57 Basin-Hopping initial guesses.")
    print(f"      Total benchmark executions: {len(instances)} instances x {len(PHASE2_CONFIGS)} configs = {len(instances)*len(PHASE2_CONFIGS)} runs.")

    # Save instance metadata for reproducibility
    instances_meta = [
        {
            "instance_idx": inst["instance_idx"],
            "case_name": inst["case_info"]["case_name"],
            "symmetry": inst["case_info"]["symmetry"],
            "dim": len(inst["case_info"]["bounds"]),
            "origin": inst["origin"],
            "trial_idx": inst["trial_idx"],
            "p0": inst["x0"].tolist(),
            "instance_seed": inst["instance_seed"]
        }
        for inst in instances
    ]
    with open(os.path.join(current_dir, "benchmark_instances.json"), "w", encoding="utf-8") as f:
        json.dump(instances_meta, f, indent=2)

    n_workers = min(16, cpu_count())
    baseline_name = "C01_Baseline_Default"

    # 3. Execute Phase 2 Across 15 Configurations
    p2_jsonl = os.path.join(current_dir, "phase2_results.jsonl")
    p2_raw, p2_by_cfg, p2_wall = execute_benchmark(PHASE2_CONFIGS, instances, n_workers, p2_jsonl)

    with open(os.path.join(current_dir, "phase2_results.json"), "w", encoding="utf-8") as f:
        json.dump(p2_raw, f, indent=2)

    # 4. Statistical Analysis
    all_stats = compute_statistics(p2_by_cfg, baseline_name, instances)

    with open(os.path.join(current_dir, "benchmark_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    overall_stats = all_stats["overall"]

    print("\n" + "=" * 110)
    print("                      PHASE 2 DUAL ANNEALING BENCHMARK SUMMARY (OVERALL: 76 INSTANCES)")
    print("=" * 110)
    print(f"{'Configuration':<42} | {'Mean SSD':<9} | {'Median':<8} | {'Diff vs Base':<16} | {'W / L / T':<10} | {'Time(s)':<8}")
    print("-" * 110)
    for name, st in overall_stats.items():
        diff_str = f"{st['paired_diff_mean']:+.3f}+/-{st['paired_diff_stderr']:.3f}"
        wlt_str = f"{st['wins']}/{st['losses']}/{st['ties']}"
        print(f"{name:<42} | {st['mean_ssd']:<9.3f} | {st['median_ssd']:<8.3f} | {diff_str:<16} | {wlt_str:<10} | {st['mean_time_sec']:<8.2f}")
    print("=" * 110)

    # 5. Write Markdown Report
    md_path = os.path.join(current_dir, "benchmark_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# YouZhiSan Dual Annealing Optimization Benchmark (Phase 2 Matrix)\n\n")
        f.write(f"- **Total Problem Instances**: {len(instances)} (19 stochastic + 57 Basin-Hopping starting points)\n")
        f.write(f"- **Configurations Tested**: {len(PHASE2_CONFIGS)} Synergistic Parameter Sets\n")
        f.write(f"- **Total Optimization Runs**: {len(instances) * len(PHASE2_CONFIGS)}\n")
        f.write(f"- **Parallel Workers**: {n_workers} CPU cores\n")
        f.write(f"- **Total Benchmark Wall Time**: {p2_wall:.2f} s ({p2_wall/60.0:.2f} min)\n\n")

        f.write("## Overall Results Table (All 76 Instances)\n\n")
        f.write("| Configuration | Mean SSD | Median SSD | Paired Diff vs Base | Win / Loss / Tie | p-value (t-test / Wilcoxon) | Avg Time (s) | Avg nfev |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for name, st in overall_stats.items():
            f.write(f"| `{name}` | {st['mean_ssd']:.3f} | {st['median_ssd']:.3f} | {st['paired_diff_mean']:+.3f} ± {st['paired_diff_stderr']:.3f} | {st['wins']} / {st['losses']} / {st['ties']} | {st['p_value_ttest']:.4f} / {st['p_value_wilcoxon']:.4f} | {st['mean_time_sec']:.2f} | {st['mean_nfev']:.0f} |\n")

        f.write("\n## Subgroup: 19 Stochastic Initial Guesses (seed=42)\n\n")
        f.write("| Configuration | Mean SSD | Median SSD | Paired Diff vs Base | Win / Loss / Tie | Avg Time (s) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for name, st in all_stats["random_init_19"].items():
            f.write(f"| `{name}` | {st['mean_ssd']:.3f} | {st['median_ssd']:.3f} | {st['paired_diff_mean']:+.3f} ± {st['paired_diff_stderr']:.3f} | {st['wins']} / {st['losses']} / {st['ties']} | {st['mean_time_sec']:.2f} |\n")

        f.write("\n## Subgroup: 57 Basin-Hopping Refined Initial Guesses\n\n")
        f.write("| Configuration | Mean SSD | Median SSD | Paired Diff vs Base | Win / Loss / Tie | Avg Time (s) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for name, st in all_stats["bh_fitted_init_57"].items():
            f.write(f"| `{name}` | {st['mean_ssd']:.3f} | {st['median_ssd']:.3f} | {st['paired_diff_mean']:+.3f} ± {st['paired_diff_stderr']:.3f} | {st['wins']} / {st['losses']} / {st['ties']} | {st['mean_time_sec']:.2f} |\n")

    print(f"\nMarkdown summary saved to: {md_path}")
    print("Benchmark completed successfully.")


if __name__ == "__main__":
    main()
