"""
Periodic Table Data for AutoAIM Studio v2.0.0

Comprehensive elemental properties used by all featurizers:
- MoleculeFeaturizer (.xyz linking)
- CrystalFeatureExtractor (CIF linking)
- FeatureEngineer (Magpie composition)

v2.0: Unified data source — all linkers derive composition from the same
elemental properties, eliminating duplication with Magpie.

Data sources:
- Atomic numbers: IUPAC
- Atomic radii (Slater): J.C. Slater, J. Chem. Phys. 1964
- Covalent radii: Cordero et al., Dalton Trans. 2008
- Electronegativity: Pauling scale
- Ionization energy: NIST
- Electron affinity: NIST
- Melting point: Wikipedia/CRC Handbook
- Thermal conductivity: Wikipedia/CRC Handbook
- Oxidation states: Greenwood & Earnshaw, Chemistry of the Elements
"""

from typing import Dict, Tuple, List, Optional


# ── Atomic Masses (IUPAC 2021, in u) ────────────────────────────
ATOMIC_MASSES: Dict[str, float] = {
    'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811,
    'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
    'S': 32.065, 'Cl': 35.453, 'K': 39.098, 'Ca': 40.078,
    'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938,
    'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.380,
    'Ga': 69.723, 'Ge': 72.640, 'As': 74.922, 'Se': 78.960, 'Br': 79.904,
    'Kr': 83.798, 'Rb': 85.468, 'Sr': 87.620, 'Y': 88.906, 'Zr': 91.224,
    'Nb': 92.906, 'Mo': 95.960, 'Tc': 98.000, 'Ru': 101.070, 'Rh': 102.906,
    'Pd': 106.420, 'Ag': 107.868, 'Cd': 112.411, 'In': 114.818,
    'Sn': 118.710, 'Sb': 121.760, 'Te': 127.600, 'I': 126.904,
    'Xe': 131.294, 'Cs': 132.905, 'Ba': 137.327, 'La': 138.905,
    'Ce': 140.116, 'Pr': 140.908, 'Nd': 144.242, 'W': 183.840,
    'Pt': 195.084, 'Au': 196.967, 'Hg': 200.590, 'Tl': 204.380,
    'Pb': 207.200, 'Bi': 208.980, 'Th': 232.038, 'U': 238.029,
}

# ── Atomic Numbers ───────────────────────────────────────────────
ATOMIC_NUMBERS: Dict[str, int] = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23,
    'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
    'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
    'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
    'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'W': 74, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83,
    'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'U': 92,
}


# ── Atomic Radii (Slater, in Angstroms) ──────────────────────────
ATOMIC_RADII: Dict[str, float] = {
    'H': 0.25, 'He': 0.31, 'Li': 1.45, 'Be': 1.05, 'B': 0.85,
    'C': 0.70, 'N': 0.65, 'O': 0.60, 'F': 0.50, 'Ne': 0.38,
    'Na': 1.80, 'Mg': 1.50, 'Al': 1.25, 'Si': 1.10, 'P': 1.00,
    'S': 1.00, 'Cl': 1.00, 'K': 2.20, 'Ca': 1.80, 'Sc': 1.60,
    'Ti': 1.40, 'V': 1.35, 'Cr': 1.40, 'Mn': 1.40, 'Fe': 1.40,
    'Co': 1.35, 'Ni': 1.35, 'Cu': 1.35, 'Zn': 1.35, 'Ga': 1.30,
    'Ge': 1.25, 'As': 1.15, 'Se': 1.15, 'Br': 1.15,
    'Rb': 2.35, 'Sr': 2.00, 'Y': 1.80, 'Zr': 1.55, 'Nb': 1.45,
    'Mo': 1.45, 'Tc': 1.35, 'Ru': 1.30, 'Rh': 1.35, 'Pd': 1.40,
    'Ag': 1.60, 'Cd': 1.55, 'In': 1.55, 'Sn': 1.45, 'Sb': 1.45,
    'Te': 1.40, 'I': 1.40,
    'Cs': 2.60, 'Ba': 2.15, 'La': 1.95, 'W': 1.35, 'Pt': 1.35,
    'Au': 1.35, 'Hg': 1.50, 'Tl': 1.90, 'Pb': 1.80, 'Bi': 1.60,
}


# ── Covalent Radii (Cordero 2008, in Angstroms) ──────────────────
COVALENT_RADII: Dict[str, float] = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39,
    'Si': 1.11, 'B': 0.84, 'Li': 1.28, 'Na': 1.66, 'Mg': 1.41,
    'Al': 1.21, 'K': 2.03, 'Ca': 1.76, 'Fe': 1.32, 'Cu': 1.32,
    'Zn': 1.22, 'Ti': 1.36, 'Ni': 1.24, 'Co': 1.26, 'Mn': 1.27,
    'Cr': 1.22, 'V': 1.25, 'Ga': 1.22, 'Ge': 1.20, 'As': 1.19,
    'Se': 1.20, 'Ag': 1.39, 'Au': 1.36, 'Pt': 1.36, 'Pd': 1.30,
    'Sn': 1.40, 'Pb': 1.44, 'Bi': 1.51, 'W': 1.36, 'Nb': 1.46,
    'Mo': 1.38, 'Zr': 1.48, 'Ta': 1.38, 'Re': 1.31, 'Os': 1.28,
    'Ir': 1.29, 'Cd': 1.44, 'In': 1.42, 'Sb': 1.39, 'Te': 1.38,
    'Y': 1.90, 'La': 1.87, 'Ce': 1.83, 'Pr': 1.82, 'Nd': 1.81,
    'Hg': 1.32, 'Tl': 1.45, 'Rb': 2.11, 'Sr': 1.92, 'Ba': 1.98,
    'Cs': 2.25, 'He': 0.28, 'Ne': 0.58, 'Ar': 1.06, 'Kr': 1.16,
    'Xe': 1.40, 'Ra': 2.15, 'Ac': 1.98, 'Th': 1.79, 'U': 1.70,
}


# ── First Ionization Energy (kJ/mol) ─────────────────────────────
IONIZATION_ENERGIES: Dict[str, float] = {
    'H': 1312.0, 'He': 2372.3, 'Li': 520.2, 'Be': 899.5, 'B': 800.6,
    'C': 1086.5, 'N': 1402.3, 'O': 1313.9, 'F': 1681.0, 'Ne': 2080.7,
    'Na': 495.8, 'Mg': 737.7, 'Al': 577.5, 'Si': 786.5, 'P': 1011.8,
    'S': 999.6, 'Cl': 1251.2, 'K': 418.8, 'Ca': 589.8, 'Sc': 633.1,
    'Ti': 658.8, 'V': 650.9, 'Cr': 652.9, 'Mn': 717.3, 'Fe': 762.5,
    'Co': 760.4, 'Ni': 737.1, 'Cu': 745.5, 'Zn': 906.4, 'Ga': 578.8,
    'Ge': 762.2, 'As': 947.0, 'Se': 941.0, 'Br': 1139.9,
    'Rb': 403.0, 'Sr': 549.5, 'Y': 600.0, 'Zr': 640.1, 'Nb': 652.1,
    'Mo': 684.3, 'Tc': 702.0, 'Ru': 710.2, 'Rh': 719.7, 'Pd': 804.4,
    'Ag': 731.0, 'Cd': 867.8, 'In': 558.3, 'Sn': 708.6, 'Sb': 830.6,
    'Te': 869.3, 'I': 1008.4,
    'Cs': 375.7, 'Ba': 502.9, 'La': 538.1, 'W': 770.0, 'Pt': 870.0,
    'Au': 890.1, 'Hg': 1007.1, 'Tl': 589.4, 'Pb': 715.6, 'Bi': 703.0,
}


# ── Electron Affinity (kJ/mol) ───────────────────────────────────
ELECTRON_AFFINITIES: Dict[str, float] = {
    'H': 72.8, 'C': 121.9, 'N': -7.0, 'O': 141.0, 'F': 328.0,
    'P': 72.0, 'S': 200.4, 'Cl': 349.0, 'Br': 324.6, 'I': 295.2,
    'Si': 134.1, 'Li': 59.6, 'Na': 52.8, 'Mg': -40.0, 'Al': 41.8,
    'K': 48.4, 'Ca': 2.37, 'Fe': 14.6, 'Cu': 118.5, 'Zn': -58.0,
    'Ga': 29.0, 'Ge': 119.0, 'As': 78.0, 'Se': 194.9,
    'Ag': 125.6, 'Au': 222.8, 'Pt': 205.3, 'O': 141.0,
    'Sn': 107.3, 'Pb': 35.2, 'Bi': 91.3, 'W': 78.6,
    'Cr': 64.3, 'Mn': -50.0, 'Co': 63.7, 'Ni': 111.7,
}


# ── Pauling Electronegativity ────────────────────────────────────
ELECTRONEGATIVITIES: Dict[str, float] = {
    'H': 2.20, 'He': 0.00, 'Li': 0.98, 'Be': 1.57, 'B': 2.04,
    'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98, 'Ne': 0.00,
    'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19,
    'S': 2.58, 'Cl': 3.16, 'K': 0.82, 'Ca': 1.00,
    'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55,
    'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65,
    'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96,
    'Kr': 3.00, 'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33,
    'Nb': 1.60, 'Mo': 2.16, 'Tc': 1.90, 'Ru': 2.20, 'Rh': 2.28,
    'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69, 'In': 1.78, 'Sn': 1.96,
    'Sb': 2.05, 'Te': 2.10, 'I': 2.66, 'Xe': 2.60,
    'Cs': 0.79, 'Ba': 0.89, 'La': 1.10, 'Ce': 1.12, 'Pr': 1.13,
    'Nd': 1.14, 'W': 2.36, 'Pt': 2.28, 'Au': 2.54, 'Hg': 2.00,
    'Tl': 1.62, 'Pb': 2.33, 'Bi': 2.02,
}


# ── Group and Period ─────────────────────────────────────────────
GROUP: Dict[str, int] = {
    'H': 1, 'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1, 'Fr': 1,
    'Be': 2, 'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2, 'Ra': 2,
    'Sc': 3, 'Y': 3, 'La': 3, 'Ac': 3,
    'Ti': 4, 'Zr': 4, 'Hf': 4,
    'V': 5, 'Nb': 5, 'Ta': 5,
    'Cr': 6, 'Mo': 6, 'W': 6,
    'Mn': 7, 'Tc': 7, 'Re': 7,
    'Fe': 8, 'Ru': 8, 'Os': 8,
    'Co': 9, 'Rh': 9, 'Ir': 9,
    'Ni': 10, 'Pd': 10, 'Pt': 10,
    'Cu': 11, 'Ag': 11, 'Au': 11,
    'Zn': 12, 'Cd': 12, 'Hg': 12,
    'B': 13, 'Al': 13, 'Ga': 13, 'In': 13, 'Tl': 13,
    'C': 14, 'Si': 14, 'Ge': 14, 'Sn': 14, 'Pb': 14,
    'N': 15, 'P': 15, 'As': 15, 'Sb': 15, 'Bi': 15,
    'O': 16, 'S': 16, 'Se': 16, 'Te': 16,
    'F': 17, 'Cl': 17, 'Br': 17, 'I': 17,
    'He': 18, 'Ne': 18, 'Ar': 18, 'Kr': 18, 'Xe': 18,
}

PERIOD: Dict[str, int] = {
    'H': 1, 'He': 1,
    'Li': 2, 'Be': 2, 'B': 2, 'C': 2, 'N': 2, 'O': 2, 'F': 2, 'Ne': 2,
    'Na': 3, 'Mg': 3, 'Al': 3, 'Si': 3, 'P': 3, 'S': 3, 'Cl': 3, 'Ar': 3,
    'K': 4, 'Ca': 4, 'Sc': 4, 'Ti': 4, 'V': 4, 'Cr': 4, 'Mn': 4,
    'Fe': 4, 'Co': 4, 'Ni': 4, 'Cu': 4, 'Zn': 4,
    'Ga': 4, 'Ge': 4, 'As': 4, 'Se': 4, 'Br': 4, 'Kr': 4,
    'Rb': 5, 'Sr': 5, 'Y': 5, 'Zr': 5, 'Nb': 5, 'Mo': 5, 'Tc': 5,
    'Ru': 5, 'Rh': 5, 'Pd': 5, 'Ag': 5, 'Cd': 5,
    'In': 5, 'Sn': 5, 'Sb': 5, 'Te': 5, 'I': 5, 'Xe': 5,
    'Cs': 6, 'Ba': 6, 'La': 6, 'Ce': 6, 'Pr': 6, 'Nd': 6,
    'W': 6, 'Pt': 6, 'Au': 6, 'Hg': 6, 'Tl': 6, 'Pb': 6, 'Bi': 6,
    'Fr': 7, 'Ra': 7, 'Ac': 7, 'Th': 7, 'U': 7,
}


# ── Melting Point (K) ────────────────────────────────────────────
MELTING_POINTS: Dict[str, float] = {
    'H': 13.99, 'He': 0.95, 'Li': 453.69, 'Be': 1560.0, 'B': 2349.0,
    'C': 3915.0, 'N': 63.15, 'O': 54.36, 'F': 53.53, 'Ne': 24.56,
    'Na': 370.87, 'Mg': 923.0, 'Al': 933.47, 'Si': 1687.0, 'P': 317.30,
    'S': 388.36, 'Cl': 171.6, 'K': 336.53, 'Ca': 1115.0,
    'Sc': 1814.0, 'Ti': 1941.0, 'V': 2183.0, 'Cr': 2180.0, 'Mn': 1519.0,
    'Fe': 1811.0, 'Co': 1768.0, 'Ni': 1728.0, 'Cu': 1357.77, 'Zn': 692.68,
    'Ga': 302.91, 'Ge': 1211.40, 'As': 1090.0, 'Se': 494.0, 'Br': 265.8,
    'Kr': 115.79, 'Rb': 312.46, 'Sr': 1050.0, 'Y': 1799.0, 'Zr': 2128.0,
    'Nb': 2750.0, 'Mo': 2896.0, 'Tc': 2430.0, 'Ru': 2607.0, 'Rh': 2237.0,
    'Pd': 1828.05, 'Ag': 1234.93, 'Cd': 594.22, 'In': 429.75,
    'Sn': 505.08, 'Sb': 903.78, 'Te': 722.66, 'I': 386.85,
    'Cs': 301.59, 'Ba': 1000.0, 'La': 1193.0, 'Ce': 1068.0,
    'W': 3695.0, 'Pt': 2041.4, 'Au': 1337.33, 'Hg': 234.32,
    'Tl': 577.0, 'Pb': 600.61, 'Bi': 544.7,
}


# ── Thermal Conductivity (W/m·K) ─────────────────────────────────
THERMAL_CONDUCTIVITIES: Dict[str, float] = {
    'H': 0.1805, 'He': 0.1513, 'Li': 84.8, 'Be': 190.0, 'B': 27.4,
    'C': 140.0, 'N': 0.02583, 'O': 0.02658, 'F': 0.0277, 'Ne': 0.0491,
    'Na': 142.0, 'Mg': 156.0, 'Al': 237.0, 'Si': 148.0, 'P': 0.236,
    'S': 0.205, 'Cl': 0.0089, 'K': 102.5, 'Ca': 201.0,
    'Sc': 15.8, 'Ti': 21.9, 'V': 30.7, 'Cr': 93.9, 'Mn': 7.81,
    'Fe': 80.4, 'Co': 100.0, 'Ni': 90.9, 'Cu': 401.0, 'Zn': 116.0,
    'Ga': 40.6, 'Ge': 60.2, 'As': 50.2, 'Se': 0.519, 'Br': 0.122,
    'Kr': 0.00943, 'Rb': 58.2, 'Sr': 35.4, 'Y': 17.2, 'Zr': 22.6,
    'Nb': 53.7, 'Mo': 138.0, 'Tc': 50.6, 'Ru': 117.0, 'Rh': 150.0,
    'Pd': 71.8, 'Ag': 429.0, 'Cd': 96.6, 'In': 81.8,
    'Sn': 66.8, 'Sb': 24.4, 'Te': 1.97, 'I': 0.449,
    'Cs': 35.9, 'Ba': 18.4, 'La': 13.4, 'W': 173.0, 'Pt': 71.6,
    'Au': 317.0, 'Hg': 8.30, 'Tl': 46.1, 'Pb': 35.3, 'Bi': 7.97,
}


# ── Common Oxidation States ──────────────────────────────────────
# Sources: Greenwood & Earnshaw, Chemistry of the Elements (2nd ed.)
# Listed in order of prevalence (most common first)
COMMON_OXIDATION_STATES: Dict[str, List[int]] = {
    # Group 1
    'H': [1, -1], 'Li': [1], 'Na': [1], 'K': [1], 'Rb': [1], 'Cs': [1],
    # Group 2
    'Be': [2], 'Mg': [2], 'Ca': [2], 'Sr': [2], 'Ba': [2], 'Ra': [2],
    # Group 3
    'Sc': [3], 'Y': [3], 'La': [3], 'Ac': [3],
    # Group 4
    'Ti': [4, 3, 2], 'Zr': [4], 'Hf': [4],
    # Group 5
    'V': [5, 4, 3, 2], 'Nb': [5, 4, 3], 'Ta': [5],
    # Group 6
    'Cr': [3, 6, 2], 'Mo': [6, 4, 5, 3], 'W': [6, 4, 5],
    # Group 7
    'Mn': [2, 4, 7, 3, 6], 'Tc': [7, 4], 'Re': [4, 7, 6],
    # Group 8
    'Fe': [3, 2], 'Ru': [4, 3, 8], 'Os': [4, 8],
    # Group 9
    'Co': [2, 3], 'Rh': [3, 4], 'Ir': [3, 4],
    # Group 10
    'Ni': [2, 3], 'Pd': [2, 4], 'Pt': [2, 4],
    # Group 11
    'Cu': [2, 1], 'Ag': [1], 'Au': [3, 1],
    # Group 12
    'Zn': [2], 'Cd': [2], 'Hg': [2, 1],
    # Group 13
    'B': [3], 'Al': [3], 'Ga': [3], 'In': [3], 'Tl': [1, 3],
    # Group 14
    'C': [4, 2, -4], 'Si': [4], 'Ge': [4, 2], 'Sn': [4, 2], 'Pb': [2, 4],
    # Group 15
    'N': [-3, 3, 5, 2, 4], 'P': [5, 3, -3], 'As': [3, 5], 'Sb': [3, 5], 'Bi': [3],
    # Group 16
    'O': [-2, -1], 'S': [-2, 4, 6], 'Se': [-2, 4, 6], 'Te': [-2, 4, 6],
    # Group 17
    'F': [-1], 'Cl': [-1, 1, 3, 5, 7], 'Br': [-1, 1, 3, 5], 'I': [-1, 1, 3, 5, 7],
    # Lanthanides
    'Ce': [3, 4], 'Pr': [3], 'Nd': [3],
    # Actinides
    'Th': [4], 'U': [6, 4],
}


# ── Helper: weighted mean and range for a property ───────────────

def weighted_mean(elements: List[str], counts: List[int],
                  prop_dict: Dict[str, float], default: float = 0.0) -> float:
    """Calculate composition-weighted mean of a property."""
    total = sum(counts)
    if total == 0:
        return default
    vals = []
    for e, c in zip(elements, counts):
        v = prop_dict.get(e)
        if v is not None:
            vals.append(v * c / total)
    return sum(vals) if vals else default


def weighted_range(elements: List[str], prop_dict: Dict[str, float],
                   default: float = 0.0) -> float:
    """Calculate range (max - min) of a property across elements."""
    vals = [prop_dict[e] for e in elements if e in prop_dict]
    return (max(vals) - min(vals)) if len(vals) >= 2 else default


def weighted_std(elements: List[str], counts: List[int],
                 prop_dict: Dict[str, float], default: float = 0.0) -> float:
    """Calculate composition-weighted standard deviation."""
    import math
    total = sum(counts)
    if total == 0 or len(elements) < 2:
        return default
    vals = []
    weights = []
    for e, c in zip(elements, counts):
        v = prop_dict.get(e)
        if v is not None:
            vals.append(v)
            weights.append(c / total)
    if not vals:
        return default
    mean = sum(v * w for v, w in zip(vals, weights))
    variance = sum(w * (v - mean) ** 2 for v, w in zip(vals, weights))
    return math.sqrt(variance) if variance > 0 else default


# ── Oxidation State Assignment ───────────────────────────────────

def assign_oxidation_states(elements: List[str], counts: List[int]) -> Dict[str, int]:
    """Assign oxidation states to each element in a compound.

    Uses known common states first, then electroneutrality fallback.
    Returns: {element_symbol: oxidation_state}
    Note: returns ONE state per unique element (most common in compound).
    """
    from collections import Counter
    import math

    total_atoms = sum(counts)
    if total_atoms == 0:
        return {}

    # Build element list with counts
    elem_count = Counter()
    for e, c in zip(elements, counts):
        elem_count[e] += c

    unique_elements = list(elem_count.keys())
    n = len(unique_elements)

    # Special case: single element (elemental form)
    if n == 1:
        return {unique_elements[0]: 0}

    # Try known states with electroneutrality constraint
    # Priority: O(-2), F(-1), H(+1), then metals by group
    assignment = {}
    remaining_charge = 0

    # First pass: assign well-known states
    for elem in unique_elements:
        states = COMMON_OXIDATION_STATES.get(elem, [])
        count = elem_count[elem]

        if elem == 'O':
            assignment['O'] = -2
            remaining_charge += (-2) * count
        elif elem == 'F':
            assignment['F'] = -1
            remaining_charge += (-1) * count
        elif elem == 'H' and 'O' in unique_elements:
            # H is +1 when bonded to non-metals (especially O)
            assignment['H'] = 1
            remaining_charge += 1 * count
        elif not states:
            # Unknown element — will handle in fallback
            continue

    # Second pass: assign remaining elements using electroneutrality
    unassigned = [e for e in unique_elements if e not in assignment]
    if unassigned:
        remaining_atoms = sum(elem_count[e] for e in unassigned)
        if remaining_atoms > 0:
            # Distribute charge equally among unassigned elements
            charge_per_atom = -remaining_charge / remaining_atoms
            for elem in unassigned:
                states = COMMON_OXIDATION_STATES.get(elem, [])
                if states:
                    # Pick the state closest to charge_per_atom
                    best_state = min(states, key=lambda s: abs(s - charge_per_atom))
                    assignment[elem] = best_state
                else:
                    # No known states — use electroneutrality directly
                    assignment[elem] = int(round(charge_per_atom))

    return assignment


def count_oxidation_states(elements: List[str], counts: List[int]) -> Dict[str, float]:
    """Count atoms in each oxidation state category.

    Returns normalized counts for: -3, -2, -1, 0, +1, +2, +3, +4, +5, +6, +7
    """
    states = assign_oxidation_states(elements, counts)
    from collections import Counter, defaultdict

    elem_count = Counter()
    for e, c in zip(elements, counts):
        elem_count[e] += c

    state_counts = Counter()
    for elem, count in elem_count.items():
        ox = states.get(elem, 0)
        state_counts[ox] += count

    total = sum(elem_count.values())
    result = {}
    for ox in range(-3, 8):
        key = f'ox_state_{ox:+d}_frac' if ox != 0 else 'ox_state_0_frac'
        result[key] = state_counts.get(ox, 0) / total if total > 0 else 0.0

    # Add summary stats
    positive = sum(v for k, v in state_counts.items() if k > 0)
    negative = sum(v for k, v in state_counts.items() if k < 0)
    result['ox_n_positive'] = positive / total if total > 0 else 0.0
    result['ox_n_negative'] = negative / total if total > 0 else 0.0
    result['ox_n_zero'] = state_counts.get(0, 0) / total if total > 0 else 0.0

    return result
