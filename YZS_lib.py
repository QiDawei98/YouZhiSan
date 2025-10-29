# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 19:14:03 2024

@author: qidaw
"""

print('*************************************************************')
print('**beta program for spectroscopy analysis with crystal field**')
print('**author: Qi Dawei                                         **')
print('**Powered by PyCrystalField                                **')
print('*************************************************************')

import PyCrystalField as cef
import io
import sys
import re
import os
import math
import numpy as np
import time
from scipy.optimize import basinhopping, dual_annealing
import matplotlib.pyplot as plt

#operator equivalent factor θ collected from literature
#α, β, γ for n = 2, 4, 6 respectively
Thet = {
    
    #Judd B R, Pryce M H L, 1957. An analysis of the absorption spectrum of praseodymium chloride[J/OL]. Proceedings of the Royal Society of London. Series A. Mathematical and Physical Sciences, 241(1226): 414-422. DOI:10.1098/rspa.1957.0136.
    ('Pr3+', '3P1'): [-1/5, 0, 0], #[α, β, γ]
    ('Pr3+', '3P2'): [1 / 15 * 1.02, 4 / (7 * 27) * 0.13, 0],
    ('Pr3+', '1D2'): [22/(15 * 21) * 0.95, 4 / (7 * 27) * 0.82, 0],
    ('Pr3+', '3F3'): [1 / 90, -1/(45 * 99), -1/(39*99)],
    ('Pr3+', '3F4'): [1/ 126 * 0.44, -1 / (45 * 77) * 1.78, 1 / (13 * 63 * 99) * (-1.16)],
    ('Pr3+', '1G4'): [-2/(11 * 35) * 0.24, -46 / (11 * 45 * 77) * (0.79), -4 / (13 * 33 * 77) * (0.71)],
    ('Pr3+', '3H4'): [-52/(25*99)*(0.98), -4/(55*99)*(1.03), 16 * 17 / (9*55*91*99)*(0.87)],
    
    
    #Judd B R, Pryce M H L, 1954. Operator equivalents and matrix elements for the excited states of rare-earth ions[J/OL]. Proceedings of the Royal Society of London. Series A. Mathematical and Physical Sciences, 227(1171): 552-560. DOI:10.1098/rspa.1955.0029.
    ('Nd3+', '4F7/2'): [2/189, 1/(21 * 45 * 11), -8 / (27 * 13 * 33 * 7)],
    ('Nd3+', '4F9/2'): [1/(6 * 18), -1/(42 * 99), 1/(126 * 39 * 33)],
    ('Nd3+', '4G5/2'): [1/(14 * 42), 13 / (77 * 42), 0],
    ('Nd3+', '4G7/2'): [26 / (3 * 49 * 225), 13 / (9 * 49 * 121), 8 / (33 * 77 * 27)],
    ('Nd3+', '4G9/2'): [7 / (4 * 33 * 99), 5 / (6 * 33 * 121), 19 / (42 * 11 * 33 * 99)],
    
    #Rabbiner, N. (1969). J. Opt. Soc. Am., JOSA 59, 588–591.
    ('Sm3+', '6H7/2'): [1.6 * 10**(-2), -2.0 * 10**(-4), -1.6 * 10**(-4)],
    ('Sm3+', '6H9/2'): [9.8 * 10**(-3), -8.3 * 10**(-5), -2.4 * 10**(-5)],
    ('Sm3+', '6H11/2'): [7.5 * 10**(-3), -2.5 * 10**(-6), -6.5 * 10**(-6)],

    

    #Lempicki, A., Samelson, H. & Brecher, C. (1968). Journal of Molecular Spectroscopy 27, 375–401.
    #('Eu3+', '7F0'): [0, 0, 0],
    ('Eu3+', '7F1'): [-1/5, 0, 0],
    ('Eu3+', '7F2'): [-11/315, -2/189, 0],
    ('Eu3+', '7F3'): [-1/135, 1/1485, -2/11583],
    ('Eu3+', '7F4'): [1/385, 23/38115, 2/33033],
    ('Eu3+', '7F5'): [1/135, 2/10395, -1/81081],
    ('Eu3+', '7F6'): [1/99, -2/16335, 1/891891],
    
    #Rabbiner, N. (1967). J. Opt. Soc. Am., JOSA 57, 217–231.
    ('Tb3+', '5D4'): [None, -1.42 * 10**(-4), -2.78 * 10**(-6)],
    ('Tb3+', '7F2'): [None, 106 * 10**(-4), 0],
    ('Tb3+', '7F3'): [None, -6.734 * 10**(-4), 172.6 * 10**(-6)],
    ('Tb3+', '7F4'): [None, -6.03 * 10**(-4), -60.5 * 10**(-6)],
    ('Tb3+', '7F5'): [None, -1.92 * 10**(-4), 12.3 * 10**(-6)],
#    ('Tb3+', '7F6'): [None, 1.22 * 10**(-4), -1.12 * 10**(-6)],     #identical with stevens52
    
    #Crosswhite, H. M. & Dieke, G. H. (1961). The Journal of Chemical Physics 35, 1535–1548.
    ('Dy3+', '6H15/2'): [-6.349 * 10 ** (-3), -5.92 * 10 ** (-5), 1.035 * 10 ** (-6)],
    ('Dy3+', '6H13/2'): [-6.838 * 10 ** (-3), -3.767 * 10 ** (-5), -1.208 * 10 ** (-6)],
    ('Dy3+', '6H11/2'): [-7.823 * 10 ** (-3), 0.269 * 10 ** (-5), -6.267 * 10 ** (-6)],
    ('Dy3+', '6H9/2'): [-10.101 * 10 ** (-3), 8.447 * 10 ** (-5), -23.459 * 10 ** (-6)],
    ('Dy3+', '6H7/2'): [-16.508 * 10 ** (-3), 20.212 * 10 ** (-5), -152.495 * 10 ** (-6)],
    ('Dy3+', '6H5/2'): [-41.270 * 10 ** (-3), -250.120 * 10 ** (-5), 0],
    ('Dy3+', '6F11/2'): [4.040 * 10 ** (-3), -6.144 * 10 ** (-5), 0.748 * 10 ** (-6)],
    ('Dy3+', '6F9/2'): [3.367 * 10 ** (-3), 8.260 * 10 ** (-5), -8.970 * 10 ** (-6)],
    ('Dy3+', '6F7/2'): [2.116 * 10 ** (-3), 36.342 * 10 ** (-5), 49.333 * 10 ** (-6)],
    ('Dy3+', '6F5/2'): [-0.635 * 10 ** (-3), 105.820 * 10 ** (-5), 0],
    ('Dy3+', '6F3/2'): [-8.889 * 10 ** (-3), 0, 0],
    
    #Erath, E. H. (1961). The Journal of Chemical Physics 34, 1985–1989.
    ('Er3+', '4I15/2'): [2.6947 * 10 ** (-3), 4.5187 * 10 ** (-5), 1.9993 * 10 ** (-6)],
    ('Er3+', '4I13/2'): [3.1414 * 10 ** (-3), 5.6546 * 10 ** (-5), 1.7793 * 10 ** (-6)],
    ('Er3+', '4I11/2'): [3.0065 * 10 ** (-3), 4.7676 * 10 ** (-5), 1.7388 * 10 ** (-6)],
    ('Er3+', '4F9/2'): [-0.0335 * 10 ** (-3), 1.7901 * 10 ** (-4), 3.7779 * 10 ** (-5)],
    ('Er3+', '4I9/2'): [-5.544 * 10 ** (-3), 1.9305 * 10 ** (-4), -8.8316 * 10 ** (-6)],
    ('Er3+', '4S3/2'): [-4.1486 * 10 ** (-2), 0, 0],
    ('Er3+', '2H11/2'): [6.6214 * 10 ** (-3), -7.1658 * 10 ** (-5), 3.7354 * 10 ** (-6)],
    ('Er3+', '4F7/2'): [-1.2232 * 10 ** (-2), -1.9906 * 10 ** (-4), 8.6970 * 10 ** (-5)],
    ('Er3+', '4F5/2'): [-7.1188 * 10 ** (-3), -5.5951 * 10 ** (-4), 0],
    ('Er3+', '4F3/2'): [-5.2393 * 10 ** (-2), 0, 0],
    ('Er3+', '2G9/2'): [-2.0067 * 10 ** (-3), 1.6796 * 10 ** (-4), 3.1415 * 10 ** (-5)],
    ('Er3+', '4G11/2'): [-7.7444 * 10 ** (-4), -1.3714 * 10 ** (-4), 2.8561 * 10 ** (-6)],
    ('Er3+', '2K15/2'): [7.3342 * 10 ** (-3), 7.4081 * 10 ** (-5), -4.0018 * 10 ** (-7)],
    ('Er3+', '4G9/2'): [5.1951 * 10 ** (-4), -8.2378 * 10 ** (-5), -2.8031 * 10 ** (-6)],
    ('Er3+', '4G7/2'): [-1.8842 * 10 ** (-3), 1.5604 * 10 ** (-4), 2.3716 * 10 ** (-5)],
    ('Er3+', '2D3/2'): [-6.4868 * 10 ** (-2), 0, 0],
    ('Er3+', '2K13/2'): [8.7419 * 10 ** (-3), 1.1104 * 10 ** (-4), 6.7514 * 10 ** (-7)],
#    ('Er3+', '2P1/2'): [0, 0, 0],
    ('Er3+', '4G5/2'): [-3.5547 * 10 ** (-3), -3.9926 * 10 ** (-3), 0],
    ('Er3+', '2G7/2'): [-1.1080 * 10 ** (-2), -3.8057 * 10 ** (-4), -5.8047 * 10 ** (-6)],

    #Wong E Y, Richman I, 1961. Analysis of the Absorption Spectrum and Zeeman Effect of Thulium Ethylsulphate[J/OL]. The Journal of Chemical Physics, 34(4): 1182-1185. DOI:10.1063/1.1731717.
    ('Tm3+', '3H6'): [1.0201 * 10 ** (-3), 1.5921 * 10 ** (-4), -5.5284 * 10 ** (-6)],
    ('Tm3+', '3H4'): [-2.1226 * 10 ** (-3), 7.2693 * 10 ** (-4), 6.8818 * 10 ** (-5)],
    ('Tm3+', '3H5'): [1.3333 * 10 ** (-2), 2.5653 * 10 ** (-4), -7.4000 * 10 ** (-6)],
    ('Tm3+', '3F4'): [1.1755 * 10 ** (-2), 4.4426 * 10 ** (-4), -7.0530 * 10 ** (-5)],
    ('Tm3+', '3F3'): [-1.1111 * 10 ** (-2), 2.2447 * 10 ** (-4), 2.5900 * 10 ** (-4)],
    ('Tm3+', '3F2'): [-4.1271 * 10 ** (-2), -6.4127 * 10 ** (-3), 0],
    ('Tm3+', '1G4'): [8.6361 * 10 ** (-3), 1.0589 * 10 ** (-3), 7.4142 * 10 ** (-5)],
    ('Tm3+', '1D2'): [-4.7656 * 10 ** (-2), 2.2859 * 10 ** (-3), 0],
    ('Tm3+', '1I6'): [2.0102 * 10 ** (-2), -2.4083 * 10 ** (-4), 2.1647 * 10 ** (-6)],
    #('Tm3+', '3P0'): [0, 0, 0],
    ('Tm3+', '3P1'): [0.2, 0, 0],
    ('Tm3+', '3P2'): [-7.2980 * 10 ** (-2), -1.13510 * 10 ** (-2), 0],
    #('Tm3+', '1S0'): [0, 0, 0],
    
    #Demirkhanyan, H. G., Demirkhanyan, G. G., Babajanyan, V. G., Kostanyan, R. B. & Kokanyan, E. P. (2008). J. Contemp. Phys. 43, 13–18.
    ('Yb3+', '2F7/2'): [2 / (7 * 9), -2 / (3 * 5 * 7 * 11), 4 / (3 * 7 * 9 * 11 * 13)],
    ('Yb3+', '2F5/2'): [2 / (5 * 7), -2 / (5 * 7 * 9), 0],
    

    #Stevens, K. W. H. (1952). Proc. Phys. Soc. A 65, 209.
    ('Ce3+', '2F5/2'): [-2./(5*7), 2./(3*3*5*7), 0],
    ('Nd3+', '4I9/2'): [-7./(3**2*11**2) , -2.**3*17/(3**3*11**3*13), -5.*17*19/(3**3*7*11**3*13**2)],
    ('Pm3+', '5I4'): [2*7./(3*5*11**2), 2.**3*7*17/(3**3*5*11**3*13), 2.**3*17*19/(3**3*7*11**2*13**2)],
    ('Sm3+', '6H5/2'): [13./(3**2*5*7) , 2.*13/(3**3*5*7*11), 0],
#    ('Gd3+', '8S7/2'): [0, 0, 0], 
    ('Tb3+', '7F6'): [-1./(3**2*11), 2./(3**3*5*11**2), -1./(3**4*7*11**2*13)],
    ('Ho3+', '5I8'): [-1./(2*3*3*5*5), -1./(2*3*5*7*11*13), -5./(3**3*7*11**2*13**2)],
#    ('Tm3+', '3H6'): [1/99, 8 / (3 * 11 * 1485), -5 / (13 * 33 * 2079), #typo in Stevens52
    
    #taken from PyCrystalField code
#    ('U4+', '3H4'): [-2.*2*13/(3*3*5*5*11), -2.*2/(3*3*5*11*11), 2.**4*17/(3**4*5*7*11**2*13)], #same as Pr3+
#    ('U3+', '4I9/2'): [-7./(3**2*11**2) , -2.**3*17/(3**3*11**3*13), -5.*17*19/(3**3*7*11**3*13**2)] #same as Nd3+
}

alt_Thet = {
    
    #Judd B R, Pryce M H L, 1954. Operator equivalents and matrix elements for the excited states of rare-earth ions[J/OL]. Proceedings of the Royal Society of London. Series A. Mathematical and Physical Sciences, 227(1171): 552-560. DOI:10.1098/rspa.1955.0029.
    ('Pr3+', '3P1'): [-1/5, 0, 0],
    ('Pr3+', '3P2'): [1/15, 0, 0],
    ('Pr3+', '1D2'): [22/(21 * 15), 4 / (27 * 7), 0],
    ('Pr3+', '1G4'): [-2 / (35 * 11), -46 / (45 * 77 * 11), -4 / (13 * 33 * 77)],
    ('Pr3+', '1I6'): [-2/99, 4/(11 * 15 * 99), -4/(13 * 33 * 77)],
    # ('Nd3+', '4F7/2'): [2/189, 1/(21 * 45 * 11), -8 / (27 * 13 * 33 * 7)],
    # ('Nd3+', '4F9/2'): [1/(6 * 18), -1/(42 * 99), 1/(126 * 39 * 33)],
    # ('Nd3+', '4G5/2'): [1/(14 * 42), 13 / (77 * 42), 0],
    # ('Nd3+', '4G7/2'): [26 / (3 * 49 * 225), 13 / (9 * 49 * 121), 8 / (33 * 77 * 27)],
    # ('Nd3+', '4G9/2'): [7 / (4 * 33 * 99), 5 / (6 * 33 * 121), 19 / (42 * 11 * 33 * 99)],
    
    
    #Koningstein, J. A. & Geusic, J. E. (1964). Phys. Rev. 136, A711–A716.
    #We can't trust this data!!!
    ('Nd3+', '4I9/2'): [-36.8162 * 10**(-3), -23.3573 * 10**(-3), -104.1088 * 10**(-3)],
    ('Nd3+', '4I11/2'): [-4.0502 * 10**(-3), -11.4408 * 10**(-3), -21.9549 * 10**(-3)],
    ('Nd3+', '4I13/2'): [-18.6096 * 10**(-3), -3.3908 * 10**(-3), -3.8778 * 10**(-3)],
    ('Nd3+', '4I15/2'): [-7.8483 * 10**(-3), -2.6371 * 10**(-3), -28.4666 * 10**(-3)],
    ('Nd3+', '4F3/2'): [151.4 * 10**(-3), 0, 0],
    ('Nd3+', '4F5/2'): [28.2323 * 10**(-3), 82.5740 * 10**(-3), 0],
    ('Nd3+', '4F9/2'): [36.5130 * 10**(-3), 10.8066 * 10**(-3), -111.3777 * 10**(-3)],
    ('Nd3+', '4S3/2'): [15.2 * 10**(-3), 0, 0],
    ('Nd3+', '4G5/2'): [6.5915 * 10**(-3), 249.2206 * 10**(-3), 0],
    ('Nd3+', '2G7/2'): [11.0249 * 10**(-3), -1.4783 * 10**(-3), 43.4223 * 10**(-3)],
    
    #Gruber, J. B. & Conway, J. G. (1960). The Journal of Chemical Physics 32, 1531–1534.
    ('Tm3+', '3P1'): [(1/5), 0, 0],
    ('Tm3+', '3P2'): [(-1/15)*1.095, (-4/(7 * 27)) * 0.629, 0],
    ('Tm3+', '1D2'): [(-22/(15 * 21))*0.678, (-4/(7 * 27)) * (-0.105), 0],
    ('Tm3+', '3F2'): [(-1/90), (1/(45 * 99)), (1/(39 * 99))],
    ('Tm3+', '3F3'): [(-1/126) * (-1.357), (1/(45*99))*(1.527), (-1/(13 * 63 * 99))*(5.945)],
    ('Tm3+', '3F4'): [(-1/126)*(-1.357), (1/(45*77))*(1.527), (-1/(13 * 33 * 21 * 99))*(0.967)],
    ('Tm3+', '1G4'): [(2/(11*35))*(1.799), (46/(11*45*77))*(0.866), (4/(13*33*77))*(0.537)],
    ('Tm3+', '1I6'): [(2/99)*0.995, (-4/(11*15*99))*(0.984), (2/(13*33*21*99))*(0.967)],
    ('Tm3+', '3H6'): [(1/99)*1.01, (8/(3*11*1485))*(0.976), (-5/(13*33*2079))*(0.986)]
}


allowed_symmetries = [
    "C1",
    "S2",
    "C2",
    "C1H",
    "C2H",
    "D2",
    "C2V",
    "D2H",
    "C4",
    "S4",
    "C4H",
    "D4",
    "C4V",
    "D2D",
    "D4H",
    "C3",
    "S6",
    "D3",
    "C3V",
    "D3D",
    "C6",
    "C3H",
    "C6H",
    "D6",
    "C6V",
    "D3H",
    "D6H",
    "T",
    "TH",
    "TD",
    "O",
    "OH",
]

cubic_symmetry = {'OH', 'O', 'TD', 'T', 'TH'}

def get_stark_levels_count(site_symmetry, J):
    '''Get the number of stark levels from site symmetry and quantum number J

    Parameters
    ----------
    Site_symmetry: str
        Example: 'T' 'C3v'
    J: int or float
    ----------
    Hänninen, P. & Härmä, H. (2011). Lanthanide Luminescence: Photophysical, 
    Analytical and Biological Aspects Berlin, Heidelberg: Springer.
    
    陈学元, 1998. 晶场对称性方法研究稀土激光晶体的光谱和磁学性能[D/OL]. 中国科学院研究生院（福建物质结构研究所）[2025-06-27]. 
    https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CDFD&dbname=CDFD9908&filename=2005152770.nh. '''
    
    assert isinstance(site_symmetry, str)
    assert isinstance(J, (float, int))
    assert site_symmetry in allowed_symmetries
    
    Cubic = {'T', 'TD', 'O', 'OH', 'TH'}
    Hexagonal = {'C3H', 'D3H', 'C6', 'C6H', 'C6V', 'D6', 'D6H'}
    Trigonal = {'C3', 'S6', 'C3V', 'D3', 'D3D'}
    Tetragonal = {'C4', 'S4', 'C4H', 'C4V', 'D4', 'D2D', 'D4H'}
#    Low = {'C1', 'Cs', 'C2', 'C2h', 'C2v', 'D2', 'D2h', 'Ci'}
    Low = {'C1', 'C1H', 'C2', 'C2H', 'C2V', 'D2', 'D2H', 'S2'}
    
    Cubic_int = {0:1, 1:1, 2:2, 3:3, 4:4, 5:4, 6:6, 7:6, 8:7}
    Hexagonal_int = {0:1, 1:2, 2:3, 3:5, 4:6, 5:7, 6:9, 7:10, 8:11}
    trig_tet_int = {0:1, 1:2, 2:4, 3:5, 4:7, 5:8, 6:10, 7:11, 8:13}
    Low_int = {0:1, 1:3, 2:5, 3:7, 4:9, 5:11, 6:13, 7:15, 8:17}
    
    Cubic_half = {0.5:1, 1.5:1, 2.5:2, 3.5:3, 4.5:3, 5.5:4, 6.5:5, 7.5:6, 8.5:6}
    Others_half = {0.5:1, 1.5:2, 2.5:3, 3.5:4, 4.5:5, 5.5:6, 6.5:7, 7.5:8, 8.5:9}
    
    if isinstance(J, int):
        if site_symmetry in Cubic:
            return Cubic_int[J]
        elif site_symmetry in Hexagonal:
            return Hexagonal_int[J]
        elif site_symmetry in Trigonal | Tetragonal:
            return trig_tet_int[J]
        elif site_symmetry in Low:
            return Low_int[J]
    elif isinstance(J, float):
        if site_symmetry in Cubic:
            for key in Cubic_half:
                if math.isclose(J, key):
                    return Cubic_half[key]
        elif site_symmetry in Hexagonal | Trigonal | Tetragonal | Low:
            for key in Others_half:
                if math.isclose(J, key):
                    return Others_half[key]
    

def PCM_Ar(ciffile, mag_ion = None, Zaxis = None, Yaxis = None, 
           crystalImage = False, NumIonNeighbors = 1, ForceImaginary = False, 
           CoordinateNumber = None, MaxDistance = None):
    
    '''Estimates A^n_m<r^n> using the point charge model.

    Options are inherited from the PyCrystalField `importCIF` function.'''
    
# =============================================================================
#   Paraneters of this function is inherited from PyCrystalField importCIF method.
#   For all parameters used in PCM_Ar:
#       included in both cef.importCIF, cef.FindPointGroupSymOps: 
#           Zaxis
#           Yaxis
#           crystalImage
#           NumIonNeighbors
#           CoordinationNumber
#           MaxDistance(maxDistance)(Not mentioned in wiki)
#       included in cef.importCIF only:
#           ciffile
#           mag_ion(used by cef_theta also)
#           ForceImaginary
#       included in cef.importCIF but ignored here:
#           ionL
#           ionS
# =============================================================================

    ##Never worked, dogshit
    # def no_theta(*args, **kwargs):
    #     return 1
    # cef.theta = no_theta

    # Get the ion name to retrieve theta
    sys.stdout = open(os.devnull, 'w') #Silence output

    #Code taken from PyCrystalField.importCIF
    cif = cef.CifFile(ciffile)
    if mag_ion == None:
        for asuc in cif.asymunitcell:
            if asuc[1].strip('3+') in ['Sm','Pm','Nd','Ce','Dy','Ho','Tm','Pr','Er','Tb','Yb']:
                mag_ion = asuc[0] #To Be used by FindPointGroupSymOps
                break
        centralion, *_ = cef.FindPointGroupSymOps(cif, ion = mag_ion, Zaxis = Zaxis,
                                           Yaxis = Yaxis, crystalImage = crystalImage,
                                           NumIonNeighbors = NumIonNeighbors,
                                           CoordinationNumber = CoordinateNumber,
                                           maxDistance = MaxDistance)
        mag_ion = None #restore to None
    else:
        #use provided mag_ion
        centralion, *_ = cef.FindPointGroupSymOps(cif, ion = mag_ion, Zaxis = Zaxis,
                                           Yaxis = Yaxis, crystalImage = crystalImage,
                                           NumIonNeighbors = NumIonNeighbors,
                                           CoordinationNumber = CoordinateNumber,
                                           maxDistance = MaxDistance)
            
    
    sys.stdout.close()
    sys.stdout = sys.__stdout__ #Restore console output 

# =============================================================================
#   The variable 'mag_ion' is used by the methods 'FindGroupSymOps' and 'importCIF'.
#   It should be provided by the user and must match the FIRST column in the CIF file,
#   which is '_atom_site_aniso_label'.
#   e.g., 'Yb1'
# 
#   The variable 'centralion' is used to retrieve the built-in theta value.
#   It should match the SECOND column in the CIF file, which is '_atom_site_aniso_type_symbol',
#   and must be formatted as 'Yb3+'.
#     
#   We get the value of centralion from mag_ion ('Yb1' to 'Yb3+')
#   because the cef.theta method requires rare-earth ions in that format.
# =============================================================================
    
# =============================================================================
#   The variable is referred to as 'MaxDistance' when used as a positional argument
#   in the 'importCIF' function.
#   In contrast, it is called 'maxDistance' when utilised in the 'FindPointGroupOps'
#   function.
# =============================================================================
    
# =============================================================================
#   PyCrystalField determines ligands based on the following priority: 
#   maxDistance, CoordinateNumber, and then NumIonNeighbors.
#   Please see in 'FindGroupSymOps':
#   if maxDistance != None:
#       pass /* set CoordinationNumber based on maxDistance */
#   if CoordinationNumber != None:
#       pass /* If CoordinationNumber exists (either specified by the user or 
#               generated in the previous step), find the ligands. */
#   else:
#       pass /* Find ligands by counting their ion neighbors. */
# =============================================================================

    
        
    #Import CIF file using the original importCIF method
    #Suppress unphysical screen output
    buffer = io.StringIO()
    sys.stdout = buffer

    imported_cif = cef.importCIF(ciffile, mag_ion = mag_ion, Zaxis = Zaxis, 
                                 Yaxis = Yaxis, crystalImage = crystalImage, 
                                 NumIonNeighbors = NumIonNeighbors,
                                 ForceImaginary = ForceImaginary,
                                 CoordinationNumber = CoordinateNumber,
                                 MaxDistance =MaxDistance)

    
    sys.stdout = sys.__stdout__
    buffer = buffer.getvalue()
    buffer = buffer.splitlines()

    #Stop printing incorret B values.
    for line in buffer:
        if re.search('Creating a point charge model', line):
            break
        else:
            print(line)

    
    print('')
    print('******************')
    print('')

# =============================================================================
#   According to PyCrystalField's logic, the treatment of magnetic ions is not
#   affected by duplicate atom definitions.
#   This logic only affects how ligands are handled, and we will follow the same 
#   approach in our implementation
# 
#   no multiply defined atoms:
#       tuple    (lig, PCM)
#   else:
#       [[Ligands1, CFLevels1], [Ligands2, CFLevels2], [Ligands3, CFLevels3]]
#       1 : original,  2 : atom position with B, b or ', 
#       3 : atom position without suffixes
# =============================================================================
    #Different branches for multiply defined atoms and magnesium ion types.
    ciffile = os.path.basename(ciffile)#Updates ciffile to only contain the filename
    if isinstance(imported_cif, tuple):
        if isinstance(imported_cif[1], cef.CFLevels):
            
            print('Estimated A^n_m<r^n> for ' + ciffile + ':')
            
            nmlabels = [re.sub(r'[B_^]', '', string) for string in imported_cif[1].BnmLabels]
            #nmlabels example:['20', '40', '43', '60', '63', '66']
            #BnmLabels example:['B_2^0', 'B_4^0', 'B_4^3', 'B_6^0', 'B_6^3', 'B_6^6']
            assert len(nmlabels) == len(imported_cif[1].B)
            for i in range(len(nmlabels)):
                n = int(nmlabels[i][0])
                m = int(nmlabels[i][1])
                print(f'    A^{n}_{m}<r^{n}>: ', end='')
                print(imported_cif[1].B[i] / cef.theta(centralion, n))     
            
        elif isinstance(imported_cif[1], cef.LS_CFLevels):
            print("Unfortunately, we only support processing of Rare Earth Elements at this time.")
        else:
            raise TypeError
    else:
        if (isinstance(imported_cif[0][1], cef.CFLevels) and
            isinstance(imported_cif[1][1], cef.CFLevels) and
            isinstance(imported_cif[2][1], cef.CFLevels)):
            
            print('Estimated A^n_m<r^n> for the original' + ciffile + ':')
            nmlabels = [re.sub(r'[B_^]', '', string) for string in imported_cif[0][1].BnmLabels]
            assert len(nmlabels) == len(imported_cif[0][1].B)
            for i in range(len(nmlabels)):
                n = int(nmlabels[i][0])
                m = int(nmlabels[i][1])
                print(f'    A^{n}_{m}<r^{n}>: ', end='')
                print(imported_cif[0][1].B[i] / cef.theta(centralion, int(nmlabels[i][0]))) 
            print('')
            
            print("Estimated A^n_m<r^n> for atom position with B, b or ' " + ':')
            nmlabels = [re.sub(r'[B_^]', '', string) for string in imported_cif[1][1].BnmLabels]
            assert len(nmlabels) == len(imported_cif[1][1].B)
            for i in range(len(nmlabels)):
                n = int(nmlabels[i][0])
                m = int(nmlabels[i][1])
                print(f'    A^{n}_{m}<r^{n}>: ', end='')
                print(imported_cif[1][1].B[i] / cef.theta(centralion, int(nmlabels[i][0]))) 
            print('')
                
            print('Estimated A^n_m<r^n> for atom position without suffixes' + ':')
            nmlabels = [re.sub(r'[B_^]', '', string) for string in imported_cif[2][1].BnmLabels]
            assert len(nmlabels) == len(imported_cif[2][1].B)
            for i in range(len(nmlabels)):
                n = int(nmlabels[i][0])
                m = int(nmlabels[i][1])
                print(f'    A^{n}_{m}<r^{n}>: ', end='')
                print(imported_cif[2][1].B[i] / cef.theta(centralion, int(nmlabels[i][0]))) 
            print('')
        
        elif(isinstance(imported_cif[0][1], cef.LS_CFLevels) and
             isinstance(imported_cif[1][1], cef.LS_CFLevels) and
             isinstance(imported_cif[2][1], cef.LS_CFLevels)):
            print("Unfortunately, we only support processing of Rare Earth Elements at this time.")
        else:
            raise TypeError
            
            
def PCM_Eigen(ciffile, mag_ion = None, Zaxis = None, Yaxis = None,
              crystalImage = False, NumIonNeighbors = 1, ForceImaginary = False,
              CoordinateNumber = None, MaxDistance = None, LaTeX = False):
    '''Estimates eigenvalues and eigenvectors using the point charge model.
    
    The LaTeX parameter controls whether the eigenvectors are output in the LaTeX format.'''
# =============================================================================
#   The logic is very similar to the PCM_Ar function. See that function for a
#   explanation.
# =============================================================================
    
    # Get the ion name to retrieve theta
    sys.stdout = open(os.devnull, 'w') #Silence output

    #Code taken from PyCrystalField.importCIF
    cif = cef.CifFile(ciffile)
    if mag_ion == None:
        for asuc in cif.asymunitcell:
            if asuc[1].strip('3+') in ['Sm','Pm','Nd','Ce','Dy','Ho','Tm','Pr','Er','Tb','Yb']:
                mag_ion = asuc[0] #To Be used by FindPointGroupSymOps
                break
        centralion, *_ = cef.FindPointGroupSymOps(cif, ion = mag_ion, Zaxis = Zaxis,
                                           Yaxis = Yaxis, crystalImage = crystalImage,
                                           NumIonNeighbors = NumIonNeighbors,
                                           CoordinationNumber = CoordinateNumber,
                                           maxDistance = MaxDistance)
        mag_ion = None #restore to None
    else:
        #use provided mag_ion
        centralion, *_ = cef.FindPointGroupSymOps(cif, ion = mag_ion, Zaxis = Zaxis,
                                           Yaxis = Yaxis, crystalImage = crystalImage,
                                           NumIonNeighbors = NumIonNeighbors,
                                           CoordinationNumber = CoordinateNumber,
                                           maxDistance = MaxDistance)
            
    
    sys.stdout.close()
    sys.stdout = sys.__stdout__ #Restore console output 
        
    #Import CIF file using the original importCIF method
    #Suppress unphysical screen output
    buffer = io.StringIO()
    sys.stdout = buffer

    imported_cif = cef.importCIF(ciffile, mag_ion = mag_ion, Zaxis = Zaxis, 
                                 Yaxis = Yaxis, crystalImage = crystalImage, 
                                 NumIonNeighbors = NumIonNeighbors,
                                 ForceImaginary = ForceImaginary,
                                 CoordinationNumber = CoordinateNumber,
                                 MaxDistance =MaxDistance)

    
    sys.stdout = sys.__stdout__
    buffer = buffer.getvalue()
    buffer = buffer.splitlines()

    #Stop printing incorret B values.
    for line in buffer:
        if re.search('Creating a point charge model', line):
            break
        else:
            print(line)

    
    print('')
    print('******************')
    print('')

    #Different branches for multiply defined atoms and magnesium ion types.
    ciffile = os.path.basename(ciffile)#Updates ciffile to only contain the filename
    if isinstance(imported_cif, tuple):
        if isinstance(imported_cif[1], cef.CFLevels):
            
            print('Estimated eigenvalue of ' + str(imported_cif[1].ion) + ' in ' + ciffile + ':')
            print(' ')
            
            Ar = []
            nmlabels = [re.sub(r'[B_^]', '', string) for string in imported_cif[1].BnmLabels]
            #nmlabels example:['20', '40', '43', '60', '63', '66']
            #BnmLabels example:['B_2^0', 'B_4^0', 'B_4^3', 'B_6^0', 'B_6^3', 'B_6^6']
            assert len(nmlabels) == len(imported_cif[1].B)    
            for i in range(len(nmlabels)):
                Ar.append(imported_cif[1].B[i] / cef.theta(centralion, int(nmlabels[i][0])))
            
            #There is no alpha value for Tb3+, so we can only perform calculation of the ground state.
            if (centralion == 'Tb3+') and any(label.startswith('2') for label in nmlabels):
                print('Multiplet:  7F6')
                imported_cif[1].diagonalize()
                if LaTeX:
                    imported_cif[1].printLaTexEigenvectors()
                else:
                    imported_cif[1].printEigenvectors()
                return
                
            
            #Iterate through Thet to calculate with all available theta values.
            for key, value in Thet.items():
                
                if key[0] == str(imported_cif[1].ion): #match ion
                    print('Multiplet:  ' + key[1])
                    
                    #Extract the J value from the term symbol.
                    m = re.search(r'\d+[A-Z](\d+\/\d+|\d+)', key[1])
                    J = m.group(1)
                    if '/' in J:
                        numerator, denominator = J.split('/')
                        J = int(numerator) / int(denominator)
                    else:
                        J = int(J)
                    
                    cef.Jion[imported_cif[1].ion][2] = J
                    
                    # Include theta in the Hamiltonian
                    B = {}
                    for i in range(len(nmlabels)):
                        if nmlabels[i].startswith('2'):
                            B['B' + nmlabels[i]] = Ar[i] * value[0]
                        elif nmlabels[i].startswith('4'):
                            B['B' + nmlabels[i]] = Ar[i] * value[1]
                        elif nmlabels[i].startswith('6'):
                            B['B' + nmlabels[i]] = Ar[i] * value[2]
                        else:
                            print(nmlabels[i])
                            raise ValueError
                    xtlvl = cef.CFLevels.Bdict(centralion, B)
                    xtlvl.diagonalize()
                    if LaTeX:
                        xtlvl.printLaTexEigenvectors()
                    else:
                        xtlvl.printEigenvectors()
                    print(' ')
                            
                            
        elif isinstance(imported_cif[1], cef.LS_CFLevels):
            print("Unfortunately, we only support processing of Rare Earth Elements at this time.")
        else:
            raise TypeError
    else:
        if (isinstance(imported_cif[0][1], cef.CFLevels) and
            isinstance(imported_cif[1][1], cef.CFLevels) and
            isinstance(imported_cif[2][1], cef.CFLevels)):
            
            ###################################################################
            
            print('Estimated eigenvalue of ' + str(imported_cif[0][1].ion) + ' in the original ' + ciffile + ':')
            print(' ')
            
            Ar = []
            nmlabels = [re.sub(r'[B_^]', '', string) for string in imported_cif[0][1].BnmLabels]
            #nmlabels example:['20', '40', '43', '60', '63', '66']
            #BnmLabels example:['B_2^0', 'B_4^0', 'B_4^3', 'B_6^0', 'B_6^3', 'B_6^6']
            assert len(nmlabels) == len(imported_cif[0][1].B)    
            for i in range(len(nmlabels)):
                Ar.append(imported_cif[0][1].B[i] / cef.theta(centralion, int(nmlabels[i][0])))
            
            #There is no alpha value for Tb3+, so we can only perform calculation of the ground state.
            if (centralion == 'Tb3+') and any(label.startswith('2') for label in nmlabels):
                print('Multiplet:  7F6')
                imported_cif[0][1].diagonalize()
                if LaTeX:
                    imported_cif[0][1].printLaTexEigenvectors()
                else:
                    imported_cif[0][1].printEigenvectors()
                return
                
            
            #Iterate through Thet to calculate with all available theta values.
            for key, value in Thet.items():
                
                if key[0] == str(imported_cif[0][1].ion): #match ion
                    print('Multiplet:  ' + key[1])
                    
                    #Extract the J value from the term symbol.
                    m = re.search(r'\d+[A-Z](\d+\/\d+|\d+)', key[1])
                    J = m.group(1)
                    if '/' in J:
                        numerator, denominator = J.split('/')
                        J = int(numerator) / int(denominator)
                    else:
                        J = int(J)
                    
                    cef.Jion[imported_cif[0][1].ion][2] = J
                    
                    # Include theta in the Hamiltonian
                    B = {}
                    for i in range(len(nmlabels)):
                        if nmlabels[i].startswith('2'):
                            B['B' + nmlabels[i]] = Ar[i] * value[0]
                        elif nmlabels[i].startswith('4'):
                            B['B' + nmlabels[i]] = Ar[i] * value[1]
                        elif nmlabels[i].startswith('6'):
                            B['B' + nmlabels[i]] = Ar[i] * value[2]
                        else:
                            print(nmlabels[i])
                            raise ValueError
                    xtlvl = cef.CFLevels.Bdict(centralion, B)
                    xtlvl.diagonalize()
                    if LaTeX:
                        xtlvl.printLaTexEigenvectors()
                    else:
                        xtlvl.printEigenvectors()
                    print(' ')
            
            
            ###################################################################
            
            print("Estimated eigenvalue of atom position with B, b or ' :")
            print(' ')
            
            Ar = []
            nmlabels = [re.sub(r'[B_^]', '', string) for string in imported_cif[1][1].BnmLabels]
            #nmlabels example:['20', '40', '43', '60', '63', '66']
            #BnmLabels example:['B_2^0', 'B_4^0', 'B_4^3', 'B_6^0', 'B_6^3', 'B_6^6']
            assert len(nmlabels) == len(imported_cif[1][1].B)    
            for i in range(len(nmlabels)):
                Ar.append(imported_cif[1][1].B[i] / cef.theta(centralion, int(nmlabels[i][0])))
            
            #There is no alpha value for Tb3+, so we can only perform calculation of the ground state.
            if (centralion == 'Tb3+') and any(label.startswith('2') for label in nmlabels):
                print('Multiplet:  7F6')
                imported_cif[1][1].diagonalize()
                if LaTeX:
                    imported_cif[1][1].printLaTexEigenvectors()
                else:
                    imported_cif[1][1].printEigenvectors()
                return
                
            
            #Iterate through Thet to calculate with all available theta values.
            for key, value in Thet.items():
                
                if key[0] == str(imported_cif[1][1].ion): #match ion
                    print('Multiplet:  ' + key[1])
                    
                    #Extract the J value from the term symbol.
                    m = re.search(r'\d+[A-Z](\d+\/\d+|\d+)', key[1])
                    J = m.group(1)
                    if '/' in J:
                        numerator, denominator = J.split('/')
                        J = int(numerator) / int(denominator)
                    else:
                        J = int(J)
                    
                    cef.Jion[imported_cif[1][1].ion][2] = J
                    
                    # Include theta in the Hamiltonian
                    B = {}
                    for i in range(len(nmlabels)):
                        if nmlabels[i].startswith('2'):
                            B['B' + nmlabels[i]] = Ar[i] * value[0]
                        elif nmlabels[i].startswith('4'):
                            B['B' + nmlabels[i]] = Ar[i] * value[1]
                        elif nmlabels[i].startswith('6'):
                            B['B' + nmlabels[i]] = Ar[i] * value[2]
                        else:
                            print(nmlabels[i])
                            raise ValueError
                    xtlvl = cef.CFLevels.Bdict(centralion, B)
                    xtlvl.diagonalize()
                    if LaTeX:
                        xtlvl.printLaTexEigenvectors()
                    else:
                        xtlvl.printEigenvectors()
                    print(' ')
                
            ###################################################################
            
            print("Estimated eigenvalue for atom position without suffixes :")
            print(' ')
            
            Ar = []
            nmlabels = [re.sub(r'[B_^]', '', string) for string in imported_cif[2][1].BnmLabels]
            #nmlabels example:['20', '40', '43', '60', '63', '66']
            #BnmLabels example:['B_2^0', 'B_4^0', 'B_4^3', 'B_6^0', 'B_6^3', 'B_6^6']
            assert len(nmlabels) == len(imported_cif[2][1].B)    
            for i in range(len(nmlabels)):
                Ar.append(imported_cif[2][1].B[i] / cef.theta(centralion, int(nmlabels[i][0])))
            
            #There is no alpha value for Tb3+, so we can only perform calculation of the ground state.
            if (centralion == 'Tb3+') and any(label.startswith('2') for label in nmlabels):
                print('Multiplet:  7F6')
                imported_cif[2][1].diagonalize()
                if LaTeX:
                    imported_cif[2][1].printLaTexEigenvectors()
                else:
                    imported_cif[2][1].printEigenvectors()
                return
                
            
            #Iterate through Thet to calculate with all available theta values.
            for key, value in Thet.items():
                
                if key[0] == str(imported_cif[2][1].ion): #match ion
                    print('Multiplet:  ' + key[1])
                    
                    #Extract the J value from the term symbol.
                    m = re.search(r'\d+[A-Z](\d+\/\d+|\d+)', key[1])
                    J = m.group(1)
                    if '/' in J:
                        numerator, denominator = J.split('/')
                        J = int(numerator) / int(denominator)
                    else:
                        J = int(J)
                    
                    cef.Jion[imported_cif[2][1].ion][2] = J
                    
                    # Include theta in the Hamiltonian
                    B = {}
                    for i in range(len(nmlabels)):
                        if nmlabels[i].startswith('2'):
                            B['B' + nmlabels[i]] = Ar[i] * value[0]
                        elif nmlabels[i].startswith('4'):
                            B['B' + nmlabels[i]] = Ar[i] * value[1]
                        elif nmlabels[i].startswith('6'):
                            B['B' + nmlabels[i]] = Ar[i] * value[2]
                        else:
                            print(nmlabels[i])
                            raise ValueError
                    xtlvl = cef.CFLevels.Bdict(centralion, B)
                    xtlvl.diagonalize()
                    if LaTeX:
                        xtlvl.printLaTexEigenvectors()
                    else:
                        xtlvl.printEigenvectors()
                    print(' ')
        
        elif(isinstance(imported_cif[0][1], cef.LS_CFLevels) and
             isinstance(imported_cif[1][1], cef.LS_CFLevels) and
             isinstance(imported_cif[2][1], cef.LS_CFLevels)):
            print("Unfortunately, we only support processing of Rare Earth Elements at this time.")
        else:
            raise TypeError

def HAM_Eigen(ion, Ardict, LaTeX = False):
    '''Displays the eigenvectors for the ground and excited states of a given Hamiltonian.
    
    ion: string, giving the RE ion to be calculated ("Yb3+", "Ho3+", etc.).
    Ardict: dictionary, giving the CEF parameters Arnm and values. 
    Example: {'Ar20':-0.42, 'Ar40': 0.02, 'Ar44':0.13}.
    The LaTeX parameter controls whether the eigenvectors are output in the LaTeX format.'''
    
    for key, value in Ardict.items():
        assert isinstance(key, str)
        assert isinstance(value, (int, float))
        assert re.match(r'^Ar\d{2}$', key)
    
    ion = ion[0] + ion[1].lower() + ion[2:]
    
    #The theta values are sourced from Hutchings(1964) and Stevens(1952) via the
    #PyCrystalField library, as our collected data does not include the necessary 
    #second-rank terms.
    if (ion == 'Tb3+') and any(term.startswith('Ar2') for term in Ardict.keys()):
        print('Ion:  Tb3+')
        print('Multiplet:  7F6')
        #Include theta in the Hamiltonian
        B = {}
        for key, value in Ardict.items():
            if key.startswith('Ar2'):
                B['B' + key[2:]] = value * cef.theta(ion = ion, n = 2)
            elif key.startswith('Ar4'):
                B['B' + key[2:]] = value * cef.theta(ion = ion, n = 4)
            elif key.startswith('Ar6'):
                B['B' + key[2:]] = value * cef.theta(ion = ion, n = 6)
        xtlvl = cef.CFLevels.Bdict(ion = ion, Bdict = B)
        print(B)
        xtlvl.diagonalize()
        if LaTeX:
            xtlvl.printLaTexEigenvectors()
        else:
            xtlvl.printEigenvectors()
        return
    
    #General Case:
    for IT, OEF in Thet.items():
        if IT[0] == ion:
            print('Ion:  ' + IT[0])
            print('Multiplet:  ' + IT[1])
            B = {}
            for trm, val in Ardict.items():
                if trm.startswith('Ar2'):
                    B['B' + trm[2:]] = val * OEF[0]
                elif trm.startswith('Ar4'):
                    B['B' + trm[2:]] = val * OEF[1]
                elif trm.startswith('Ar6'):
                    B['B' + trm[2:]] = val * OEF[2]
                
            print(B)
            
            m = re.search(r'\d+[A-Z](\d+\/\d+|\d+)', IT[1])
            J = m.group(1)
            if '/' in J:
                numerator, denominator = J.split('/')
                J = int(numerator) / int(denominator)
            else:
                J = int(J)
            cef.Jion[ion][2] = J
            
            xtlvl = cef.CFLevels.Bdict(ion = ion, Bdict = B)
            xtlvl.diagonalize()
            if LaTeX:
                xtlvl.printLaTexEigenvectors()
            else:
                xtlvl.printEigenvectors()
    
    # With alternative theta value
    if ion in ['Pr3+', 'Nd3+', 'Tm3+']:
        print('=========================================')
        print('**** Results with alternative theta: ****')
        print('=========================================')
        for IT, OEF in alt_Thet.items():
            if IT[0] == ion:
                print('Ion:  ' + IT[0])
                print('Multiplet:  ' + IT[1])
                B = {}
                for trm, val in Ardict.items():
                    if trm.startswith('Ar2'):
                        B['B' + trm[2:]] = val * OEF[0]
                    elif trm.startswith('Ar4'):
                        B['B' + trm[2:]] = val * OEF[1]
                    elif trm.startswith('Ar6'):
                        B['B' + trm[2:]] = val * OEF[2]
                    
                print(B)
                
                m = re.search(r'\d+[A-Z](\d+\/\d+|\d+)', IT[1])
                J = m.group(1)
                if '/' in J:
                    numerator, denominator = J.split('/')
                    J = int(numerator) / int(denominator)
                else:
                    J = int(J)
                cef.Jion[ion][2] = J
                
                xtlvl = cef.CFLevels.Bdict(ion = ion, Bdict = B)
                xtlvl.diagonalize()
                if LaTeX:
                    xtlvl.printLaTexEigenvectors()
                else:
                    xtlvl.printEigenvectors()    

def allowed_cfp(site_symmetry):
    '''Gets the non-zero crystal field parameters for a given site symmetry.
    
    site_symmetry should be a string included in allowed_symmetries'''
    
    assert site_symmetry in allowed_symmetries
    
    #Cubic case is for completeness only; it's never used.
    if site_symmetry in {'OH', 'O', 'TD'}:
        return {'AR40':None, 'AR44':None, 'AR60':None, 'AR64':None}
    elif site_symmetry in {'T', 'TH'}:
        return {'AR40':None, 'AR44':None, 'AR60':None, 'AR64':None, 'AR62':None, 'AR66':None}
    #Katsuhiko, T., Hisatomo, H. & Akira, Y. (2013). Journal of the Physical Society of Japan
    #https://doi.org/10.1143/JPSJ.70.1190.
    
    #Goremychkin, E. A., Osborn, R., Bauer, E. D., Maple, M. B., Frederick, N. A., Yuhasz, W. M.,
    #Woodward, F. M. & Lynn, J. W. (2004). Phys. Rev. Lett. 93, 157003.
    
    #hexagonal
    elif site_symmetry in {'C6', 'C3H', 'C6H'}:
        return {'AR20':None, 'AR40':None, 'AR60':None, 'AR66':None, 'AR6-6':None}
    elif site_symmetry in {'C6V', 'D6', 'D3H', 'D6H'}:
        return {'AR20':None, 'AR40':None, 'AR60':None, 'AR66':None}
    #Rudowicz, C. (1986). Chemical Physics 102, 437–443.
    
    #trigonal
    elif site_symmetry in {'C3V', 'D3', 'D3D'}:
        return {'AR20':None, 'AR40':None, 'AR43':None, 'AR60':None, 'AR63':None, 'AR66':None}
    elif site_symmetry in {'C3', 'S6'}:
        return {'AR20':None, 'AR40':None, 'AR43':None, 'AR60':None, 'AR63':None, 'AR6-3':None,
                'AR66':None, 'AR6-6':None}
    #(Rudowicz, 1986)
    
    #tetragonal
    elif site_symmetry in {'S4', 'C4', 'C4H'}:
        return {'AR20':None, 'AR40':None, 'AR44':None, 'AR60':None, 'AR64':None, 'AR6-4':None}
    elif site_symmetry in {'C4V', 'D2D', 'D4H', 'D4'}:
        return {'AR20':None, 'AR40':None, 'AR44':None, 'AR60':None, 'AR64':None}
    #Rudowicz, C. (1985). Chemical Physics 97, 43–50.
    
    #orthorhombic
    elif site_symmetry in {'C2V', 'D2', 'D2H'}:
        return {'AR20':None, 'AR22':None, 'AR40':None, 'AR42':None, 'AR44':None, 
                'AR60':None, 'AR62':None, 'AR64':None, 'AR66':None}
    #Rudowicz, C. & Bramley, R. (1985). The Journal of Chemical Physics 83, 5192–5197.
    
    #monoclinic
    elif site_symmetry in {'C2', 'C1H', 'C2H'}:
        return {'AR20':None, 'AR22':None, 'AR40':None, 'AR42':None, 'AR4-2':None,
                'AR44':None, 'AR4-4':None,
                'AR60':None, 'AR62':None, 'AR6-2':None, 'AR64':None, 'AR6-4':None,
                'AR66':None, 'AR6-6':None}
    #Rudowicz, C. (1986). The Journal of Chemical Physics 84, 5045–5058.
    
    #triclinic
    elif site_symmetry in {'C1', 'S2'}:
        return {'AR20':None, 'AR22':None, 'AR2-2':None, 'AR40':None, 'AR42':None,
                'AR4-2':None, 'AR44':None, 'AR4-4':None, 'AR60':None, 'AR62':None,
                'AR6-2':None, 'AR64':None, 'AR6-4':None, 'AR66':None, 'AR6-6':None}
    #Mulak, J. (2003). Physica B: Condensed Matter 337, 173–179.

def reconstruct_and_run(p_ind, constraints, core_func, *extra_args):
    """Entirely generated by Google's Gemini.
    
    A wrapper to handle dependent parameters for an optimizer.

    This function acts as a bridge between a scipy optimizer and an objective
    function. The optimizer works with a reduced set of independent parameters,
    and this function reconstructs the full parameter vector based on defined
    constraints before calling the actual objective function.

    Args:
        p_ind (np.ndarray): The array of independent parameters that the
            optimizer is varying.
        constraints (dict): A dictionary defining the dependent parameters.
            Format: {dependent_idx: (independent_idx, ratio)}.
        core_func (callable): The main objective function that accepts the
            full parameter vector as its first argument.
        *extra_args: Any additional arguments (e.g., x_data, y_data) that
            need to be passed on to the `core_func`.

    Returns:
        float: The scalar value returned by `core_func`, which the
        optimizer will attempt to minimize.
    """
    # Calculate total_vars internally, making the function self-contained
    total_vars = len(p_ind) + len(constraints)
    
    p_full = np.zeros(total_vars)
    # This is a condensed way to get the independent indices
    ind_indices = sorted(list(set(range(total_vars)) - set(constraints.keys())))
    p_full[ind_indices] = p_ind
    for dep, (ind, ratio) in constraints.items():
        p_full[dep] = p_full[ind] * ratio
        
    # Pass the reconstructed vector and any extra args to the real objective
    return core_func(p_full, *extra_args)


def THNRG(ion, Ar_dict, p, term_to_fit, alt_thet_enabled = False):
    '''
    Calculate eigenvalues from a given crystal field hamiltonian and returns a THNRG dictionary.

    Parameters
    ----------
    ion : str
    
    Ar_dict : dict
        Example: {'AR20': None, 'AR40': None, 'AR60': None, 'AR44': None, 'AR64': None}
    
    p : np.ndarray (list is also okay)
        Values for each crystal field parameters.
    term_to_fit : list
    
    alt_thet_enabled : bool, optional
        Controls whether to use the alternative theta value. The default is False.

    Returns 
    ----------
    a THNRG dictionary.
    '''

    assert len(Ar_dict) == len(p)
    assert isinstance(Ar_dict, dict)
    assert isinstance(ion, str)
    assert isinstance(term_to_fit, list)
    assert isinstance(alt_thet_enabled, (bool, type(None)))
    
    Ar_dict = {key: value for key, value in zip(Ar_dict.keys(), p)}
    
    THNRG = {}
    
    ion = ion[0] + ion[1].lower() + ion[2:]
    
    if not alt_thet_enabled:
        active_theta = Thet
    else:
        active_theta = alt_Thet
        
    for IT, OEF in active_theta.items():
        if IT[0] == ion and IT[1] in term_to_fit:
            B = {}
            for trm, val in Ar_dict.items():
                if trm.startswith('AR2'):
                    B['B' + trm[2:]] = val * OEF[0]
                elif trm.startswith('AR4'):
                    B['B' + trm[2:]] = val * OEF[1]
                elif trm.startswith('AR6'):
                    B['B' + trm[2:]] = val * OEF[2]
                
           # print(B)
            
            m = re.search(r'\d+[A-Z](\d+\/\d+|\d+)', IT[1])
            J = m.group(1)
            if '/' in J:
                numerator, denominator = J.split('/')
                J = int(numerator) / int(denominator)
            else:
                J = int(J)
            cef.Jion[ion][2] = J
            
            xtlvl = cef.CFLevels.Bdict(ion = ion, Bdict = B)
            xtlvl.diagonalize()
            
            # np.unique will also sort the resulting array.
            THNRG[IT[1]] = np.unique([round (levels, 9) for levels in xtlvl.eigenvalues])

    return THNRG

def SSD(EXNRG, THNRG):
    """
    Calculates sum of squared differences

    Parameters
    ----------
    EXNRG : dict
    THNRG : dict

    Returns
    -------
    float

    """
    
    SSD = 0.0
    for key in THNRG:
  #      assert len(EXNRG[key]) == len(THNRG[key])  
        if len(EXNRG[key]) != len(THNRG[key]):
            # if sys.platform == "win32":
            #     import winsound
            #     winsound.Beep(440, 100)
            return float('inf')
        # For the case with no missing energy
        if all(isinstance(item, float) for item in EXNRG[key]):
            EXNRG[key].sort()
            THNRG[key].sort()
            for i in range(len(EXNRG[key])):
                SSD += (EXNRG[key][i] - THNRG[key][i]) ** 2
        # For the case when some energies are missing.
        else:
            for i in range(len(EXNRG[key])):
                THNRG[key].sort()
                if isinstance(EXNRG[key][i], float):
#                    baseline_index = i
                    baseline = EXNRG[key][i]
                    baseline_th = THNRG[key][i]
                    break
            assert baseline == 0 
            assert isinstance(baseline_th, float)
            for i in range(len(EXNRG[key])):
                THNRG[key][i] = THNRG[key][i] - baseline_th
                if not isinstance(EXNRG[key][i], float):
                    continue
                EXNRG[key][i] = EXNRG[key][i] - baseline
            for i in range(len(EXNRG[key])):
                if isinstance(EXNRG[key][i], float):
                    SSD += (EXNRG[key][i] - THNRG[key][i]) ** 2
    return SSD

def core_func(p, EXNRG, ion, Ar_dict, term_to_fit, alt_thet_enabled = False):
    return SSD(EXNRG, THNRG(ion, Ar_dict, p, term_to_fit, alt_thet_enabled))
                
def worker(run_duration_minutes, individual_bounds, constraints, core_func, EXNRG, ion, Ar_dict, term_to_fit, alt_thet_enabled):
    
    '''
    Conducts global optimisation for optimal crystal field parameters, designed to be run in parallel.
    
    run_duration_minutes, individual_bounds: These are user-specified parameters.
    Note that `individual_bounds` will be pre-processed before being 
    passed to this function.
    
    constraints: Defines parameter dependencies as used by the `reconstruct_and_run`
    function. For example: {1: (0, 5), 3: (2, -21)}
    
    core_func: self-explanatory
    
    ion, Ar_dict, alt_thet_enabled: These parameters are constant for the duration of the 
    search iteration. They are passed to `reconstruct_and_run` via *extra_args.
    '''
    
    
    sys.stdout = open(os.devnull, 'w')
    
    min_difference = float('inf')
    mc_Ar = None
    
    start_time = time.time()
    run_duration_seconds = run_duration_minutes * 60
    
    while time.time() - start_time < run_duration_seconds:
        indices_to_remove = set(constraints.keys())
        
        filtered_bounds = [bound for i, bound in enumerate(individual_bounds) if i not in indices_to_remove]
        
        Ar_array = np.array([np.random.uniform(low, high) for low, high in filtered_bounds])
        
        difference = reconstruct_and_run(
            Ar_array,          # p_ind
            constraints,       # constraints
            core_func,         # core_func
            EXNRG,             # start of *extra_args
            ion,               
            Ar_dict,           
            term_to_fit,
            alt_thet_enabled   
        )
        
        if difference < min_difference:
            mc_Ar = Ar_array
            min_difference = difference
            
    return min_difference, mc_Ar
    
    assert isinstance(mc_Ar, np.ndarray) and mc_Ar.ndim == 1
            
    class BoundedStep:
        """
        Entirely generated by Google's Gemini.
        
        A `take_step` callable for basin-hopping that takes a boundary-scaled 
        random step, clipping the result to enforce the search area.
        """
        def __init__(self, bounds):
            self.bounds = np.array(bounds)
            self.step_sizes = 0.1 * (self.bounds[:, 1] - self.bounds[:, 0])

        def __call__(self, x):
            x_new = x + np.random.uniform(-self.step_sizes, self.step_sizes, size=x.shape)
            np.clip(x_new, self.bounds[:, 0], self.bounds[:, 1], out=x_new)
            return x_new

    bh_result = basinhopping(
        func=lambda p: reconstruct_and_run(
            p, constraints, core_func, EXNRG, ion, Ar_dict, term_to_fit, alt_thet_enabled
        ),
        x0=mc_Ar,
        minimizer_kwargs={
            'method': 'L-BFGS-B',
            'bounds': filtered_bounds,
        },
        take_step=BoundedStep(filtered_bounds),
#       niter = 1000
    )
    
    da_result = dual_annealing(
        func=lambda p: reconstruct_and_run(
            p, constraints, core_func, EXNRG, ion, Ar_dict, term_to_fit, alt_thet_enabled
        ),
        bounds=filtered_bounds,
        x0=bh_result.x,
#       maxiter = 10000,
#       visit = 2.9,
#       maxfun = 1e10,
        minimizer_kwargs={
            "method": "L-BFGS-B",
            "bounds": filtered_bounds
        }
    )
      
    return da_result.fun, da_result.x
            
        
def create_energy_level_plot(THNRG, EXNRG):
    '''
    Μodified from Gemini.

    '''

    # Ensure the terms are processed in a consistent order
    spectral_terms = sorted(THNRG.keys())
    num_terms = len(spectral_terms)

    # --- Calculate global Y-axis limits for all subplots ---
    global_all_energies = []
    all_values = list(THNRG.values()) + list(EXNRG.values())
    for energy_list in all_values:
        global_all_energies.extend([e for e in energy_list if e is not None])

    y_limits = None
    if global_all_energies:
        min_e, max_e = min(global_all_energies), max(global_all_energies)
        margin = (max_e - min_e) * 0.05 if max_e > min_e else 1
        y_limits = (min_e - margin, max_e + margin)

    # --- Setup the plot grid ---
    # Arrange subplots in a grid to keep the figure reasonably proportioned
    ncols = 3 if num_terms > 4 else 2 if num_terms > 1 else 1
    nrows = (num_terms + ncols - 1) // ncols
    subplot_dim = 5.0 # Dimension in inches for a single square subplot

    # Create the figure and axes objects using an object-oriented approach
    # The figsize is set to create square subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * subplot_dim, nrows * subplot_dim),
                                 squeeze=False)

    # Flatten the 2D array of axes for easy iteration
    axes = axes.flatten()

    # --- Plot data for each spectral term ---
    for i, term in enumerate(spectral_terms):
        ax = axes[i]
        th_energies = THNRG.get(term, [])
        ex_energies = EXNRG.get(term, [])

        # --- Configure the appearance of the current subplot (ax) ---
        # Format the term symbol for the title using LaTeX for super/subscripts
        if len(term) >= 3 and term[0].isdigit() and term[1].isalpha():
            multiplicity = term[0]
            orbital = term[1]
            total_j = term[2:]
            # Example: '4I15/2' becomes '$^{4}\mathrm{I}_{15/2}$'
            formatted_title = f'$^{{{multiplicity}}}\\mathrm{{{orbital}}}_{{{{{total_j}}}}}$'
        else:
            formatted_title = term # Fallback for any unexpected formats

        ax.set_title(formatted_title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Energy (meV)')

        # Set x-axis ticks and labels for theoretical and experimental sides
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Theoretical', 'Experimental'])
        ax.tick_params(axis='x', length=0) # Hide x-axis tick marks
        ax.set_xlim(0.5, 2.5)

        # --- Plot individual energy levels ---
        level_width = 0.6
        th_x, ex_x = 1, 2

        for th_energy, ex_energy in zip(th_energies, ex_energies):
            # Plot the theoretical energy level line
            ax.plot([th_x - level_width / 2, th_x + level_width / 2],
                    [th_energy, th_energy], color='black', linewidth=1)

            # Plot experimental and connecting lines if data exists
            if ex_energy is not None:
                # Plot the experimental energy level line
                ax.plot([ex_x - level_width / 2, ex_x + level_width / 2],
                        [ex_energy, ex_energy], color='black', linewidth=1)

                # Plot the light grey dashed line connecting the two
                ax.plot([th_x + level_width / 2, ex_x - level_width / 2],
                        [th_energy, ex_energy], color='gainsboro', linestyle='--', linewidth=1)

        # Set y-axis limits to the pre-calculated global range
        if y_limits:
            ax.set_ylim(y_limits)

    # --- Final figure adjustments ---
    # Hide any unused subplots
    for i in range(num_terms, len(axes)):
        axes[i].axis('off')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Display the plot
    plt.show()    
    
    
            