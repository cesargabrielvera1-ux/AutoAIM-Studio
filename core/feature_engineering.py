"""Feature engineering for materials data, including Magpie and Matminer integration."""

import re
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils.logger import LoggerMixin


class CompositionParser:
    """Parse chemical composition formulas."""
    
    @staticmethod
    def parse_formula(formula: str) -> Dict[str, float]:
        """Parse chemical formula into element composition.
        
        Args:
            formula: Chemical formula (e.g., 'Fe2O3', 'SiO2')
            
        Returns:
            Dictionary mapping element symbols to their fractions
        """
        if not formula or not isinstance(formula, str):
            return {}
        
        formula = formula.strip()
        
        # Handle complex formulas with parentheses
        composition = defaultdict(float)
        
        # Simple regex to match elements and their counts
        pattern = r'([A-Z][a-z]?)([0-9]*\.?[0-9]*)'
        matches = re.findall(pattern, formula)
        
        for element, count in matches:
            if count == '':
                count = 1.0
            else:
                count = float(count)
            composition[element] += count
        
        # Normalize to fractions
        total = sum(composition.values())
        if total > 0:
            composition = {k: v / total for k, v in composition.items()}
        
        return dict(composition)
    
    @staticmethod
    def is_valid_formula(formula: str) -> bool:
        """Check if string is a valid chemical formula.
        
        Args:
            formula: String to check
            
        Returns:
            True if valid formula
        """
        if not formula or not isinstance(formula, str):
            return False
        
        formula = formula.strip()
        
        # Basic pattern: starts with capital letter, contains valid elements
        pattern = r'^([A-Z][a-z]?[0-9]*\.?[0-9]*)+$'
        return bool(re.match(pattern, formula))


class MagpieFeatures(BaseEstimator, TransformerMixin, LoggerMixin):
    """Generate Magpie-style features from chemical compositions."""
    
    # Extended element properties inspired by Magpie dataset
    # Includes: atomic, physical, electronic, and thermal properties
    ELEMENT_PROPERTIES = {
        'H': {'atomic_number': 1, 'atomic_mass': 1.008, 'melting_point': 14.01, 'boiling_point': 20.28,
              'group': 1, 'period': 1, 'electronegativity': 2.20, 'covalent_radius': 31, 'atomic_radius': 53,
              'ionization_energy': 1312.0, 'electron_affinity': 72.8, 'density': 0.0899, 'thermal_conductivity': 0.1805},
        'Li': {'atomic_number': 3, 'atomic_mass': 6.941, 'melting_point': 453.69, 'boiling_point': 1615.0,
               'group': 1, 'period': 2, 'electronegativity': 0.98, 'covalent_radius': 128, 'atomic_radius': 167,
               'ionization_energy': 520.2, 'electron_affinity': 59.6, 'density': 0.534, 'thermal_conductivity': 84.8},
        'Be': {'atomic_number': 4, 'atomic_mass': 9.012, 'melting_point': 1560.0, 'boiling_point': 2742.0,
               'group': 2, 'period': 2, 'electronegativity': 1.57, 'covalent_radius': 96, 'atomic_radius': 112,
               'ionization_energy': 899.5, 'electron_affinity': 0.0, 'density': 1.848, 'thermal_conductivity': 200.0},
        'B': {'atomic_number': 5, 'atomic_mass': 10.811, 'melting_point': 2349.0, 'boiling_point': 4200.0,
              'group': 13, 'period': 2, 'electronegativity': 2.04, 'covalent_radius': 84, 'atomic_radius': 87,
              'ionization_energy': 800.6, 'electron_affinity': 26.7, 'density': 2.34, 'thermal_conductivity': 27.4},
        'C': {'atomic_number': 6, 'atomic_mass': 12.011, 'melting_point': 3915.0, 'boiling_point': 3915.0,
              'group': 14, 'period': 2, 'electronegativity': 2.55, 'covalent_radius': 76, 'atomic_radius': 67,
              'ionization_energy': 1086.5, 'electron_affinity': 121.9, 'density': 2.267, 'thermal_conductivity': 140.0},
        'N': {'atomic_number': 7, 'atomic_mass': 14.007, 'melting_point': 63.15, 'boiling_point': 77.36,
              'group': 15, 'period': 2, 'electronegativity': 3.04, 'covalent_radius': 71, 'atomic_radius': 56,
              'ionization_energy': 1402.3, 'electron_affinity': -7.0, 'density': 1.251, 'thermal_conductivity': 0.0258},
        'O': {'atomic_number': 8, 'atomic_mass': 15.999, 'melting_point': 54.36, 'boiling_point': 90.20,
              'group': 16, 'period': 2, 'electronegativity': 3.44, 'covalent_radius': 66, 'atomic_radius': 48,
              'ionization_energy': 1313.9, 'electron_affinity': 141.0, 'density': 1.429, 'thermal_conductivity': 0.0265},
        'F': {'atomic_number': 9, 'atomic_mass': 18.998, 'melting_point': 53.53, 'boiling_point': 85.03,
              'group': 17, 'period': 2, 'electronegativity': 3.98, 'covalent_radius': 57, 'atomic_radius': 42,
              'ionization_energy': 1681.0, 'electron_affinity': 328.0, 'density': 1.696, 'thermal_conductivity': 0.0277},
        'Na': {'atomic_number': 11, 'atomic_mass': 22.990, 'melting_point': 370.87, 'boiling_point': 1156.0,
               'group': 1, 'period': 3, 'electronegativity': 0.93, 'covalent_radius': 166, 'atomic_radius': 190,
               'ionization_energy': 495.8, 'electron_affinity': 52.8, 'density': 0.971, 'thermal_conductivity': 141.0},
        'Mg': {'atomic_number': 12, 'atomic_mass': 24.305, 'melting_point': 923.0, 'boiling_point': 1363.0,
               'group': 2, 'period': 3, 'electronegativity': 1.31, 'covalent_radius': 141, 'atomic_radius': 145,
               'ionization_energy': 737.7, 'electron_affinity': 0.0, 'density': 1.738, 'thermal_conductivity': 156.0},
        'Al': {'atomic_number': 13, 'atomic_mass': 26.982, 'melting_point': 933.47, 'boiling_point': 2792.0,
               'group': 13, 'period': 3, 'electronegativity': 1.61, 'covalent_radius': 121, 'atomic_radius': 118,
               'ionization_energy': 577.5, 'electron_affinity': 41.8, 'density': 2.698, 'thermal_conductivity': 237.0},
        'Si': {'atomic_number': 14, 'atomic_mass': 28.086, 'melting_point': 1687.0, 'boiling_point': 3265.0,
               'group': 14, 'period': 3, 'electronegativity': 1.90, 'covalent_radius': 111, 'atomic_radius': 111,
               'ionization_energy': 786.5, 'electron_affinity': 134.1, 'density': 2.329, 'thermal_conductivity': 149.0},
        'P': {'atomic_number': 15, 'atomic_mass': 30.974, 'melting_point': 317.3, 'boiling_point': 550.0,
              'group': 15, 'period': 3, 'electronegativity': 2.19, 'covalent_radius': 107, 'atomic_radius': 98,
              'ionization_energy': 1011.8, 'electron_affinity': 72.0, 'density': 1.82, 'thermal_conductivity': 0.236},
        'S': {'atomic_number': 16, 'atomic_mass': 32.065, 'melting_point': 388.36, 'boiling_point': 717.8,
              'group': 16, 'period': 3, 'electronegativity': 2.58, 'covalent_radius': 105, 'atomic_radius': 88,
              'ionization_energy': 999.6, 'electron_affinity': 200.4, 'density': 2.067, 'thermal_conductivity': 0.205},
        'Cl': {'atomic_number': 17, 'atomic_mass': 35.453, 'melting_point': 171.6, 'boiling_point': 239.11,
               'group': 17, 'period': 3, 'electronegativity': 3.16, 'covalent_radius': 102, 'atomic_radius': 79,
               'ionization_energy': 1251.2, 'electron_affinity': 349.0, 'density': 3.214, 'thermal_conductivity': 0.0089},
        'K': {'atomic_number': 19, 'atomic_mass': 39.098, 'melting_point': 336.53, 'boiling_point': 1032.0,
              'group': 1, 'period': 4, 'electronegativity': 0.82, 'covalent_radius': 203, 'atomic_radius': 243,
              'ionization_energy': 418.8, 'electron_affinity': 48.4, 'density': 0.862, 'thermal_conductivity': 102.0},
        'Ca': {'atomic_number': 20, 'atomic_mass': 40.078, 'melting_point': 1115.0, 'boiling_point': 1757.0,
               'group': 2, 'period': 4, 'electronegativity': 1.00, 'covalent_radius': 176, 'atomic_radius': 194,
               'ionization_energy': 589.8, 'electron_affinity': 2.37, 'density': 1.550, 'thermal_conductivity': 201.0},
        'Sc': {'atomic_number': 21, 'atomic_mass': 44.956, 'melting_point': 1814.0, 'boiling_point': 3109.0,
               'group': 3, 'period': 4, 'electronegativity': 1.36, 'covalent_radius': 170, 'atomic_radius': 184,
               'ionization_energy': 633.1, 'electron_affinity': 18.1, 'density': 2.985, 'thermal_conductivity': 15.8},
        'Ti': {'atomic_number': 22, 'atomic_mass': 47.867, 'melting_point': 1941.0, 'boiling_point': 3560.0,
               'group': 4, 'period': 4, 'electronegativity': 1.54, 'covalent_radius': 160, 'atomic_radius': 176,
               'ionization_energy': 658.8, 'electron_affinity': 7.6, 'density': 4.507, 'thermal_conductivity': 21.9},
        'V': {'atomic_number': 23, 'atomic_mass': 50.942, 'melting_point': 2183.0, 'boiling_point': 3680.0,
              'group': 5, 'period': 4, 'electronegativity': 1.63, 'covalent_radius': 153, 'atomic_radius': 171,
              'ionization_energy': 650.9, 'electron_affinity': 50.6, 'density': 6.110, 'thermal_conductivity': 30.7},
        'Cr': {'atomic_number': 24, 'atomic_mass': 51.996, 'melting_point': 2180.0, 'boiling_point': 2944.0,
               'group': 6, 'period': 4, 'electronegativity': 1.66, 'covalent_radius': 139, 'atomic_radius': 166,
               'ionization_energy': 652.9, 'electron_affinity': 64.3, 'density': 7.140, 'thermal_conductivity': 93.9},
        'Mn': {'atomic_number': 25, 'atomic_mass': 54.938, 'melting_point': 1519.0, 'boiling_point': 2334.0,
               'group': 7, 'period': 4, 'electronegativity': 1.55, 'covalent_radius': 139, 'atomic_radius': 161,
               'ionization_energy': 717.3, 'electron_affinity': 0.0, 'density': 7.470, 'thermal_conductivity': 7.8},
        'Fe': {'atomic_number': 26, 'atomic_mass': 55.845, 'melting_point': 1811.0, 'boiling_point': 3134.0,
               'group': 8, 'period': 4, 'electronegativity': 1.83, 'covalent_radius': 132, 'atomic_radius': 156,
               'ionization_energy': 762.5, 'electron_affinity': 15.7, 'density': 7.874, 'thermal_conductivity': 80.4},
        'Co': {'atomic_number': 27, 'atomic_mass': 58.933, 'melting_point': 1768.0, 'boiling_point': 3200.0,
               'group': 9, 'period': 4, 'electronegativity': 1.88, 'covalent_radius': 126, 'atomic_radius': 152,
               'ionization_energy': 760.4, 'electron_affinity': 63.7, 'density': 8.900, 'thermal_conductivity': 100.0},
        'Ni': {'atomic_number': 28, 'atomic_mass': 58.693, 'melting_point': 1728.0, 'boiling_point': 3186.0,
               'group': 10, 'period': 4, 'electronegativity': 1.91, 'covalent_radius': 124, 'atomic_radius': 149,
               'ionization_energy': 737.1, 'electron_affinity': 111.6, 'density': 8.908, 'thermal_conductivity': 90.9},
        'Cu': {'atomic_number': 29, 'atomic_mass': 63.546, 'melting_point': 1357.77, 'boiling_point': 2835.0,
               'group': 11, 'period': 4, 'electronegativity': 1.90, 'covalent_radius': 132, 'atomic_radius': 145,
               'ionization_energy': 745.5, 'electron_affinity': 118.4, 'density': 8.960, 'thermal_conductivity': 401.0},
        'Zn': {'atomic_number': 30, 'atomic_mass': 65.38, 'melting_point': 692.68, 'boiling_point': 1180.0,
               'group': 12, 'period': 4, 'electronegativity': 1.65, 'covalent_radius': 122, 'atomic_radius': 142,
               'ionization_energy': 906.4, 'electron_affinity': 0.0, 'density': 7.134, 'thermal_conductivity': 116.0},
        'Ga': {'atomic_number': 31, 'atomic_mass': 69.723, 'melting_point': 302.91, 'boiling_point': 2477.0,
               'group': 13, 'period': 4, 'electronegativity': 1.81, 'covalent_radius': 122, 'atomic_radius': 136,
               'ionization_energy': 578.8, 'electron_affinity': 28.9, 'density': 5.907, 'thermal_conductivity': 40.6},
        'Ge': {'atomic_number': 32, 'atomic_mass': 72.64, 'melting_point': 1211.4, 'boiling_point': 3106.0,
               'group': 14, 'period': 4, 'electronegativity': 2.01, 'covalent_radius': 120, 'atomic_radius': 125,
               'ionization_energy': 762.2, 'electron_affinity': 118.9, 'density': 5.323, 'thermal_conductivity': 60.2},
        'As': {'atomic_number': 33, 'atomic_mass': 74.922, 'melting_point': 1090.0, 'boiling_point': 887.0,
               'group': 15, 'period': 4, 'electronegativity': 2.18, 'covalent_radius': 119, 'atomic_radius': 114,
               'ionization_energy': 947.0, 'electron_affinity': 78.5, 'density': 5.776, 'thermal_conductivity': 50.2},
        'Se': {'atomic_number': 34, 'atomic_mass': 78.96, 'melting_point': 494.0, 'boiling_point': 958.0,
               'group': 16, 'period': 4, 'electronegativity': 2.55, 'covalent_radius': 120, 'atomic_radius': 103,
               'ionization_energy': 941.0, 'electron_affinity': 194.9, 'density': 4.809, 'thermal_conductivity': 0.519},
        'Br': {'atomic_number': 35, 'atomic_mass': 79.904, 'melting_point': 265.8, 'boiling_point': 332.0,
               'group': 17, 'period': 4, 'electronegativity': 2.96, 'covalent_radius': 120, 'atomic_radius': 94,
               'ionization_energy': 1139.9, 'electron_affinity': 324.6, 'density': 3.122, 'thermal_conductivity': 0.122},
        'Rb': {'atomic_number': 37, 'atomic_mass': 85.468, 'melting_point': 312.46, 'boiling_point': 961.0,
               'group': 1, 'period': 5, 'electronegativity': 0.82, 'covalent_radius': 220, 'atomic_radius': 265,
               'ionization_energy': 403.0, 'electron_affinity': 46.9, 'density': 1.532, 'thermal_conductivity': 58.2},
        'Sr': {'atomic_number': 38, 'atomic_mass': 87.62, 'melting_point': 1050.0, 'boiling_point': 1655.0,
               'group': 2, 'period': 5, 'electronegativity': 0.95, 'covalent_radius': 195, 'atomic_radius': 219,
               'ionization_energy': 549.5, 'electron_affinity': 5.03, 'density': 2.630, 'thermal_conductivity': 35.3},
        'Y': {'atomic_number': 39, 'atomic_mass': 88.906, 'melting_point': 1799.0, 'boiling_point': 3609.0,
              'group': 3, 'period': 5, 'electronegativity': 1.22, 'covalent_radius': 190, 'atomic_radius': 212,
              'ionization_energy': 600.0, 'electron_affinity': 29.6, 'density': 4.472, 'thermal_conductivity': 17.2},
        'Zr': {'atomic_number': 40, 'atomic_mass': 91.224, 'melting_point': 2128.0, 'boiling_point': 4682.0,
               'group': 4, 'period': 5, 'electronegativity': 1.33, 'covalent_radius': 175, 'atomic_radius': 206,
               'ionization_energy': 640.1, 'electron_affinity': 41.1, 'density': 6.511, 'thermal_conductivity': 22.6},
        'Nb': {'atomic_number': 41, 'atomic_mass': 92.906, 'melting_point': 2750.0, 'boiling_point': 5017.0,
               'group': 5, 'period': 5, 'electronegativity': 1.6, 'covalent_radius': 164, 'atomic_radius': 198,
               'ionization_energy': 652.1, 'electron_affinity': 86.1, 'density': 8.570, 'thermal_conductivity': 53.7},
        'Mo': {'atomic_number': 42, 'atomic_mass': 95.96, 'melting_point': 2896.0, 'boiling_point': 4912.0,
               'group': 6, 'period': 5, 'electronegativity': 2.16, 'covalent_radius': 154, 'atomic_radius': 190,
               'ionization_energy': 684.3, 'electron_affinity': 71.9, 'density': 10.280, 'thermal_conductivity': 138.0},
        'Tc': {'atomic_number': 43, 'atomic_mass': 98.0, 'melting_point': 2430.0, 'boiling_point': 4538.0,
               'group': 7, 'period': 5, 'electronegativity': 1.9, 'covalent_radius': 147, 'atomic_radius': 183,
               'ionization_energy': 702.0, 'electron_affinity': 53.0, 'density': 11.500, 'thermal_conductivity': 50.6},
        'Ru': {'atomic_number': 44, 'atomic_mass': 101.07, 'melting_point': 2607.0, 'boiling_point': 4423.0,
               'group': 8, 'period': 5, 'electronegativity': 2.2, 'covalent_radius': 146, 'atomic_radius': 178,
               'ionization_energy': 710.2, 'electron_affinity': 100.9, 'density': 12.370, 'thermal_conductivity': 117.0},
        'Rh': {'atomic_number': 45, 'atomic_mass': 102.906, 'melting_point': 2237.0, 'boiling_point': 3968.0,
               'group': 9, 'period': 5, 'electronegativity': 2.28, 'covalent_radius': 142, 'atomic_radius': 173,
               'ionization_energy': 719.7, 'electron_affinity': 110.2, 'density': 12.450, 'thermal_conductivity': 150.0},
        'Pd': {'atomic_number': 46, 'atomic_mass': 106.42, 'melting_point': 1828.05, 'boiling_point': 3236.0,
               'group': 10, 'period': 5, 'electronegativity': 2.20, 'covalent_radius': 139, 'atomic_radius': 169,
               'ionization_energy': 804.4, 'electron_affinity': 53.7, 'density': 12.023, 'thermal_conductivity': 71.8},
        'Ag': {'atomic_number': 47, 'atomic_mass': 107.868, 'melting_point': 1234.93, 'boiling_point': 2435.0,
               'group': 11, 'period': 5, 'electronegativity': 1.93, 'covalent_radius': 145, 'atomic_radius': 165,
               'ionization_energy': 731.0, 'electron_affinity': 125.6, 'density': 10.490, 'thermal_conductivity': 429.0},
        'Cd': {'atomic_number': 48, 'atomic_mass': 112.411, 'melting_point': 594.22, 'boiling_point': 1040.0,
               'group': 12, 'period': 5, 'electronegativity': 1.69, 'covalent_radius': 144, 'atomic_radius': 161,
               'ionization_energy': 867.8, 'electron_affinity': 0.0, 'density': 8.650, 'thermal_conductivity': 96.8},
        'In': {'atomic_number': 49, 'atomic_mass': 114.818, 'melting_point': 429.75, 'boiling_point': 2345.0,
               'group': 13, 'period': 5, 'electronegativity': 1.78, 'covalent_radius': 142, 'atomic_radius': 156,
               'ionization_energy': 558.3, 'electron_affinity': 28.9, 'density': 7.310, 'thermal_conductivity': 81.8},
        'Sn': {'atomic_number': 50, 'atomic_mass': 118.71, 'melting_point': 505.08, 'boiling_point': 2875.0,
               'group': 14, 'period': 5, 'electronegativity': 1.96, 'covalent_radius': 139, 'atomic_radius': 145,
               'ionization_energy': 708.6, 'electron_affinity': 107.3, 'density': 7.287, 'thermal_conductivity': 66.8},
        'Sb': {'atomic_number': 51, 'atomic_mass': 121.76, 'melting_point': 903.78, 'boiling_point': 1860.0,
               'group': 15, 'period': 5, 'electronegativity': 2.05, 'covalent_radius': 139, 'atomic_radius': 133,
               'ionization_energy': 834.0, 'electron_affinity': 101.1, 'density': 6.685, 'thermal_conductivity': 24.4},
        'Te': {'atomic_number': 52, 'atomic_mass': 127.6, 'melting_point': 722.66, 'boiling_point': 1261.0,
               'group': 16, 'period': 5, 'electronegativity': 2.1, 'covalent_radius': 138, 'atomic_radius': 123,
               'ionization_energy': 869.3, 'electron_affinity': 190.2, 'density': 6.232, 'thermal_conductivity': 1.97},
        'I': {'atomic_number': 53, 'atomic_mass': 126.904, 'melting_point': 386.85, 'boiling_point': 457.4,
              'group': 17, 'period': 5, 'electronegativity': 2.66, 'covalent_radius': 139, 'atomic_radius': 115,
              'ionization_energy': 1008.4, 'electron_affinity': 295.2, 'density': 4.930, 'thermal_conductivity': 0.449},
        'Cs': {'atomic_number': 55, 'atomic_mass': 132.905, 'melting_point': 301.59, 'boiling_point': 944.0,
               'group': 1, 'period': 6, 'electronegativity': 0.79, 'covalent_radius': 244, 'atomic_radius': 298,
               'ionization_energy': 375.7, 'electron_affinity': 45.5, 'density': 1.873, 'thermal_conductivity': 35.9},
        'Ba': {'atomic_number': 56, 'atomic_mass': 137.327, 'melting_point': 1000.0, 'boiling_point': 2170.0,
               'group': 2, 'period': 6, 'electronegativity': 0.89, 'covalent_radius': 215, 'atomic_radius': 253,
               'ionization_energy': 502.9, 'electron_affinity': 13.95, 'density': 3.510, 'thermal_conductivity': 18.4},
        'La': {'atomic_number': 57, 'atomic_mass': 138.905, 'melting_point': 1193.0, 'boiling_point': 3737.0,
               'group': 3, 'period': 6, 'electronegativity': 1.10, 'covalent_radius': 207, 'atomic_radius': 169,
               'ionization_energy': 538.1, 'electron_affinity': 48.0, 'density': 6.150, 'thermal_conductivity': 13.5},
        'Hf': {'atomic_number': 72, 'atomic_mass': 178.49, 'melting_point': 2506.0, 'boiling_point': 4876.0,
               'group': 4, 'period': 6, 'electronegativity': 1.3, 'covalent_radius': 175, 'atomic_radius': 208,
               'ionization_energy': 658.5, 'electron_affinity': 17.2, 'density': 13.310, 'thermal_conductivity': 23.0},
        'Ta': {'atomic_number': 73, 'atomic_mass': 180.948, 'melting_point': 3290.0, 'boiling_point': 5731.0,
               'group': 5, 'period': 6, 'electronegativity': 1.5, 'covalent_radius': 170, 'atomic_radius': 200,
               'ionization_energy': 761.0, 'electron_affinity': 31.0, 'density': 16.650, 'thermal_conductivity': 57.5},
        'W': {'atomic_number': 74, 'atomic_mass': 183.84, 'melting_point': 3695.0, 'boiling_point': 5828.0,
              'group': 6, 'period': 6, 'electronegativity': 2.36, 'covalent_radius': 162, 'atomic_radius': 193,
              'ionization_energy': 770.0, 'electron_affinity': 78.6, 'density': 19.250, 'thermal_conductivity': 173.0},
        'Re': {'atomic_number': 75, 'atomic_mass': 186.207, 'melting_point': 3459.0, 'boiling_point': 5869.0,
               'group': 7, 'period': 6, 'electronegativity': 1.9, 'covalent_radius': 151, 'atomic_radius': 188,
               'ionization_energy': 760.0, 'electron_affinity': 14.5, 'density': 21.020, 'thermal_conductivity': 47.9},
        'Os': {'atomic_number': 76, 'atomic_mass': 190.23, 'melting_point': 3306.0, 'boiling_point': 5285.0,
               'group': 8, 'period': 6, 'electronegativity': 2.2, 'covalent_radius': 144, 'atomic_radius': 185,
               'ionization_energy': 840.0, 'electron_affinity': 106.1, 'density': 22.590, 'thermal_conductivity': 87.6},
        'Ir': {'atomic_number': 77, 'atomic_mass': 192.217, 'melting_point': 2719.0, 'boiling_point': 4701.0,
               'group': 9, 'period': 6, 'electronegativity': 2.20, 'covalent_radius': 141, 'atomic_radius': 180,
               'ionization_energy': 880.0, 'electron_affinity': 150.9, 'density': 22.560, 'thermal_conductivity': 147.0},
        'Pt': {'atomic_number': 78, 'atomic_mass': 195.084, 'melting_point': 2041.4, 'boiling_point': 3825.0,
               'group': 10, 'period': 6, 'electronegativity': 2.28, 'covalent_radius': 136, 'atomic_radius': 177,
               'ionization_energy': 870.0, 'electron_affinity': 205.3, 'density': 21.450, 'thermal_conductivity': 71.6},
        'Au': {'atomic_number': 79, 'atomic_mass': 196.967, 'melting_point': 1337.33, 'boiling_point': 3129.0,
               'group': 11, 'period': 6, 'electronegativity': 2.54, 'covalent_radius': 136, 'atomic_radius': 174,
               'ionization_energy': 890.1, 'electron_affinity': 222.8, 'density': 19.300, 'thermal_conductivity': 317.0},
        'Hg': {'atomic_number': 80, 'atomic_mass': 200.59, 'melting_point': 234.32, 'boiling_point': 629.88,
               'group': 12, 'period': 6, 'electronegativity': 2.00, 'covalent_radius': 132, 'atomic_radius': 171,
               'ionization_energy': 1007.1, 'electron_affinity': 0.0, 'density': 13.534, 'thermal_conductivity': 8.30},
        'Tl': {'atomic_number': 81, 'atomic_mass': 204.383, 'melting_point': 577.0, 'boiling_point': 1746.0,
               'group': 13, 'period': 6, 'electronegativity': 1.62, 'covalent_radius': 145, 'atomic_radius': 156,
               'ionization_energy': 589.4, 'electron_affinity': 19.2, 'density': 11.850, 'thermal_conductivity': 46.1},
        'Pb': {'atomic_number': 82, 'atomic_mass': 207.2, 'melting_point': 600.61, 'boiling_point': 2022.0,
               'group': 14, 'period': 6, 'electronegativity': 2.33, 'covalent_radius': 146, 'atomic_radius': 154,
               'ionization_energy': 715.6, 'electron_affinity': 35.1, 'density': 11.340, 'thermal_conductivity': 35.3},
        'Bi': {'atomic_number': 83, 'atomic_mass': 208.98, 'melting_point': 544.7, 'boiling_point': 1837.0,
               'group': 15, 'period': 6, 'electronegativity': 2.02, 'covalent_radius': 148, 'atomic_radius': 143,
               'ionization_energy': 703.0, 'electron_affinity': 91.2, 'density': 9.780, 'thermal_conductivity': 7.97},
        'Po': {'atomic_number': 84, 'atomic_mass': 209.0, 'melting_point': 527.0, 'boiling_point': 1235.0,
               'group': 16, 'period': 6, 'electronegativity': 2.0, 'covalent_radius': 140, 'atomic_radius': 135,
               'ionization_energy': 812.1, 'electron_affinity': 183.3, 'density': 9.196, 'thermal_conductivity': 0.20},
        'At': {'atomic_number': 85, 'atomic_mass': 210.0, 'melting_point': 575.0, 'boiling_point': 610.0,
               'group': 17, 'period': 6, 'electronegativity': 2.2, 'covalent_radius': 150, 'atomic_radius': 127,
               'ionization_energy': 899.0, 'electron_affinity': 270.0, 'density': 9.30, 'thermal_conductivity': 1.7},
    }
    
    def __init__(self, composition_column: str = 'composition'):
        """Initialize Magpie feature generator.
        
        Args:
            composition_column: Name of column containing chemical formulas
        """
        self.composition_column = composition_column
        self.parser = CompositionParser()
    
    def fit(self, X, y=None):
        """Fit transformer (no-op)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform compositions to Magpie features.
        
        Args:
            X: DataFrame with composition column
            
        Returns:
            DataFrame with Magpie features
        """
        if self.composition_column not in X.columns:
            raise ValueError(f"Column '{self.composition_column}' not found in data")
        
        compositions = X[self.composition_column]
        features_list = []
        
        for formula in compositions:
            features = self._compute_features(formula)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # FIX: Asegurar que todas las columnas sean numericas (float)
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').astype(float)
        
        # Add prefix to column names (using generic "comp_" prefix instead of "magpie_")
        features_df.columns = [f"comp_{col}" for col in features_df.columns]
        
        return features_df
    
    def _compute_features(self, formula: str) -> Dict[str, float]:
        """Compute Magpie features for a single formula.
        
        Args:
            formula: Chemical formula
            
        Returns:
            Dictionary of features
        """
        composition = self.parser.parse_formula(formula)
        
        if not composition:
            # Return zeros if parsing fails
            return {name: 0.0 for name in self._get_feature_names()}
        
        features = {}
        
        # Get property values for each element
        property_values = defaultdict(list)
        weights = []
        
        for element, fraction in composition.items():
            if element in self.ELEMENT_PROPERTIES:
                props = self.ELEMENT_PROPERTIES[element]
                weights.append(fraction)
                
                for prop_name, prop_value in props.items():
                    property_values[prop_name].append(prop_value)
        
        if not weights:
            return {name: 0.0 for name in self._get_feature_names()}
        
        weights = np.array(weights)
        weights /= weights.sum()  # Normalize
        
        # Compute weighted statistics for each property
        for prop_name, values in property_values.items():
            values = np.array(values)
            
            # Weighted mean
            features[f'{prop_name}_mean'] = np.average(values, weights=weights)
            
            # Weighted std
            if len(values) > 1:
                features[f'{prop_name}_std'] = np.sqrt(
                    np.average((values - features[f'{prop_name}_mean'])**2, weights=weights)
                )
            else:
                features[f'{prop_name}_std'] = 0.0
            
            # Min and max
            features[f'{prop_name}_min'] = values.min()
            features[f'{prop_name}_max'] = values.max()
            
            # Range
            features[f'{prop_name}_range'] = values.max() - values.min()
            
            # Additional statistics for more features
            if len(values) > 1:
                # Weighted median (using interpolation)
                sorted_indices = np.argsort(values)
                sorted_values = values[sorted_indices]
                sorted_weights = weights[sorted_indices]
                cumsum = np.cumsum(sorted_weights)
                median_idx = np.searchsorted(cumsum, 0.5)
                features[f'{prop_name}_median'] = sorted_values[min(median_idx, len(values)-1)]
                
                # Percentiles
                features[f'{prop_name}_p25'] = np.percentile(values, 25)
                features[f'{prop_name}_p75'] = np.percentile(values, 75)
            else:
                features[f'{prop_name}_median'] = values[0]
                features[f'{prop_name}_p25'] = values[0]
                features[f'{prop_name}_p75'] = values[0]
        
        # Additional derived features
        features['n_elements'] = float(len(composition))
        features['fractional_entropy'] = float(self._compute_entropy(weights))
        
        return features
    
    def _compute_entropy(self, fractions: np.ndarray) -> float:
        """Compute Shannon entropy of composition.
        
        Args:
            fractions: Array of element fractions
            
        Returns:
            Shannon entropy
        """
        # Filter out zero fractions
        fractions = fractions[fractions > 0]
        
        if len(fractions) <= 1:
            return 0.0
        
        return -np.sum(fractions * np.log(fractions))
    
    def _get_feature_names(self) -> List[str]:
        """Get list of feature names.
        
        Returns:
            List of feature names (13 properties × 8 stats + 2 derived = 106 features)
        """
        properties = [
            'atomic_number', 'atomic_mass', 'melting_point', 'boiling_point',
            'group', 'period', 'electronegativity', 'covalent_radius', 'atomic_radius',
            'ionization_energy', 'electron_affinity', 'density', 'thermal_conductivity'
        ]
        stats = ['mean', 'std', 'min', 'max', 'range', 'median', 'p25', 'p75']
        
        names = []
        for prop in properties:
            for stat in stats:
                names.append(f'{prop}_{stat}')
        
        names.extend(['n_elements', 'fractional_entropy'])
        return names
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return [f"comp_{name}" for name in self._get_feature_names()]


class CompositionFeatures(BaseEstimator, TransformerMixin, LoggerMixin):
    """Generate features from chemical compositions using Matminer."""
    
    def __init__(
        self,
        composition_column: str = 'composition',
        use_magpie: bool = True,
        use_matminer: bool = False,  # FIX: Deshabilitado - requiere pymatgen
        use_oxidation_states: bool = False
    ):
        """Initialize composition feature generator.
        
        Args:
            composition_column: Name of column with chemical formulas
            use_magpie: Whether to use Magpie features (implementacion propia)
            use_matminer: Whether to use Matminer features (NO USAR - requiere pymatgen)
            use_oxidation_states: Whether to include oxidation state features
        """
        self.composition_column = composition_column
        self.use_magpie = use_magpie
        self.use_matminer = False  # FIX: Forzar a False para evitar pymatgen
        self.use_oxidation_states = use_oxidation_states
        
        self._magpie = None
        self._matminer_featurizer = None
    
    def fit(self, X, y=None):
        """Fit transformer."""
        # FIX: Solo usar implementacion propia de Magpie (no pymatgen)
        if self.use_magpie:
            self._magpie = MagpieFeatures(self.composition_column)
            self._magpie.fit(X, y)
            self.logger.info("Usando implementacion propia de Magpie (106 descriptores)")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform compositions to features.
        
        Args:
            X: DataFrame with composition column
            
        Returns:
            DataFrame with composition features
        """
        feature_dfs = []
        
        if self.use_magpie and self._magpie:
            magpie_features = self._magpie.transform(X)
            feature_dfs.append(magpie_features)
        
        if self.use_oxidation_states:
            ox_features = self._compute_oxidation_features(X)
            feature_dfs.append(ox_features)
        
        if feature_dfs:
            return pd.concat(feature_dfs, axis=1)
        else:
            return pd.DataFrame(index=X.index)
    
    def _compute_oxidation_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute oxidation state related features.
        
        Args:
            X: DataFrame with composition column
            
        Returns:
            DataFrame with oxidation features
        """
        # Simplified oxidation state features
        parser = CompositionParser()
        
        features = []
        for formula in X[self.composition_column]:
            composition = parser.parse_formula(formula)
            
            # Count metal and non-metal elements (simplified)
            metals = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba',
                     'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                     'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                     'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                     'Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi']
            
            n_metals = sum(1 for elem in composition if elem in metals)
            n_nonmetals = len(composition) - n_metals
            
            features.append({
                'ox_n_metals': n_metals,
                'ox_n_nonmetals': n_nonmetals,
                'ox_metal_ratio': n_metals / len(composition) if composition else 0,
            })
        
        return pd.DataFrame(features, index=X.index)
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        names = []
        
        if self.use_magpie and self._magpie:
            names.extend(self._magpie.get_feature_names_out())
        
        if self.use_matminer and self._matminer_featurizer:
            names.extend([f"mm_{col}" for col in self._matminer_featurizer.feature_labels()])
        
        if self.use_oxidation_states:
            names.extend(['ox_n_metals', 'ox_n_nonmetals', 'ox_metal_ratio'])
        
        return names


class FeatureEngineer(LoggerMixin):
    """Main feature engineering class."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self._transformers = {}
        self._feature_names = []
    
    def add_composition_features(
        self,
        data: pd.DataFrame,
        composition_column: str,
        use_magpie: bool = True,
        use_matminer: bool = False
    ) -> pd.DataFrame:
        """Add composition-based features to dataset.
        
        Args:
            data: Input DataFrame
            composition_column: Column with chemical formulas
            use_magpie: Whether to use Magpie features
            use_matminer: Whether to use Matminer features
            
        Returns:
            DataFrame with added features (original data + new features)
        """
        if composition_column not in data.columns:
            raise ValueError(f"Column '{composition_column}' not found in data")
        
        transformer = CompositionFeatures(
            composition_column=composition_column,
            use_magpie=use_magpie,
            use_matminer=use_matminer
        )
        
        transformer.fit(data)
        comp_features = transformer.transform(data)
        
        self._transformers[f'composition_{composition_column}'] = transformer
        
        # FIX: Concatenar features nuevas CON el dataset original completo
        # (manteniendo todas las columnas originales incluyendo la de composicion)
        result = pd.concat([data, comp_features], axis=1)
        
        self.logger.info(f"Added {comp_features.shape[1]} composition features. "
                        f"Original: {data.shape[1]} cols, New: {result.shape[1]} cols")
        
        return result
    
    def add_polynomial_features(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        degree: int = 2,
        interaction_only: bool = True
    ) -> pd.DataFrame:
        """Add polynomial features.
        
        Args:
            data: Input DataFrame
            columns: Columns to use (None = all numeric)
            degree: Polynomial degree
            interaction_only: Whether to include only interaction terms
            
        Returns:
            DataFrame with added features
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        
        poly_data = poly.fit_transform(data[columns])
        feature_names = poly.get_feature_names_out(columns)
        
        poly_df = pd.DataFrame(poly_data, columns=feature_names, index=data.index)
        
        # Remove original columns to avoid duplication
        poly_df = poly_df.drop(columns=columns, errors='ignore')
        
        self._transformers['polynomial'] = poly
        
        result = pd.concat([data, poly_df], axis=1)
        
        self.logger.info(f"Added {poly_df.shape[1]} polynomial features")
        
        return result
    
    def add_statistical_features(
        self,
        data: pd.DataFrame,
        group_column: str,
        agg_column: str,
        aggregations: List[str] = None
    ) -> pd.DataFrame:
        """Add statistical features based on groupby.
        
        Args:
            data: Input DataFrame
            group_column: Column to group by
            agg_column: Column to aggregate
            aggregations: List of aggregation functions
            
        Returns:
            DataFrame with added features
        """
        if aggregations is None:
            aggregations = ['mean', 'std', 'min', 'max', 'count']
        
        grouped = data.groupby(group_column)[agg_column].agg(aggregations)
        grouped.columns = [f'{agg_column}_{group_column}_{agg}' for agg in grouped.columns]
        
        result = data.merge(grouped, left_on=group_column, right_index=True, how='left')
        
        self.logger.info(f"Added {len(aggregations)} statistical features")
        
        return result
    
    # ------------------------------------------------------------------
    # Crystal structure features (v1.3.0 — NO matminer, pymatgen only)
    # ------------------------------------------------------------------
    
    def add_crystal_structure_features(
        self,
        data: pd.DataFrame,
        structure_column: str = 'structure'
    ) -> pd.DataFrame:
        """Add crystallographic features from pymatgen Structure objects.
        
        Extracts ~25 numeric descriptors from crystal structures using only
        pymatgen (no matminer). Output feeds directly into the AutoAIM
        training pipeline.
        
        Args:
            data: DataFrame containing a column with pymatgen Structure objects
            structure_column: Name of column with Structure objects
            
        Returns:
            DataFrame with added crystal features (original data + new features)
        """
        if structure_column not in data.columns:
            raise ValueError(f"Column '{structure_column}' not found in data")
        
        try:
            from .crystal_structure import CrystalStructureFeaturizer
        except ImportError as e:
            self.logger.error(f"Cannot import CrystalStructureFeaturizer: {e}")
            return data
        
        featurizer = CrystalStructureFeaturizer()
        result = featurizer.featurize_dataframe(data, structure_column)
        
        self._transformers['crystal_structure'] = featurizer
        
        self.logger.info(
            f"Added crystal structure features. "
            f"Shape: {data.shape} -> {result.shape}"
        )
        
        return result
    
    def featurize_structures(
        self,
        structures: List[Tuple[str, 'Structure']],
        target_values: Optional[Dict[str, float]] = None,
        target_column: str = 'target'
    ) -> pd.DataFrame:
        """Convert a list of pymatgen Structures to a training-ready DataFrame.
        
        This is the main entry point for crystal structure workflows in v1.3.0.
        Loads structures, extracts crystallographic features, and optionally
        attaches target values to produce a DataFrame ready for DataManager.
        
        Args:
            structures: List of (name, Structure) tuples
            target_values: Optional dict mapping structure_name -> target value
            target_column: Name for the target column
            
        Returns:
            DataFrame with crystal features + target, ready for DataManager
        """
        from .crystal_structure import CrystalStructureFeaturizer
        
        featurizer = CrystalStructureFeaturizer()
        features_df = featurizer.featurize_structures(structures)
        
        if features_df.empty:
            raise ValueError("No features extracted from structures")
        
        # Keep only numeric columns
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df = features_df[numeric_cols].astype(float)
        
        # Add target values if provided
        if target_values and 'structure_name' in features_df.columns:
            features_df[target_column] = features_df['structure_name'].map(target_values)
            n_missing = features_df[target_column].isna().sum()
            if n_missing > 0:
                self.logger.warning(
                    f"{n_missing} structures have no target value and will be dropped"
                )
                features_df = features_df.dropna(subset=[target_column])
        
        self.logger.info(
            f"Structure dataset: {len(features_df)} samples, "
            f"{len(features_df.columns)} features"
        )
        
        return features_df
