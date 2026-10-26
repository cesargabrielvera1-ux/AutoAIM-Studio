"""
Tests for app/core/periodic_data.py

Validates the periodic data dictionaries and helper functions
that underpin all featurizers in AutoAIM Studio.

v2.0: 100% Python pure. No external dependencies beyond numpy.
"""

import math

import numpy as np
import pytest

from app.core.periodic_data import (
    # Dictionaries (keys are chemical symbol strings)
    ATOMIC_NUMBERS,
    ATOMIC_MASSES,
    ATOMIC_RADII,
    COVALENT_RADII,
    ELECTRONEGATIVITIES,
    IONIZATION_ENERGIES,
    ELECTRON_AFFINITIES,
    MELTING_POINTS,
    COMMON_OXIDATION_STATES,
    GROUP,
    PERIOD,
    THERMAL_CONDUCTIVITIES,
    # Functions
    weighted_mean,
    weighted_range,
    weighted_std,
    count_oxidation_states,
    assign_oxidation_states,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. Dictionary integrity tests
# ──────────────────────────────────────────────────────────────────────────

class TestDictionaryIntegrity:
    """All dictionaries must be consistent, complete, and valid."""

    def test_all_dictionaries_have_valid_element_keys(self):
        """Every key in every dictionary must be a valid element symbol (1-2 letters, capitalized)."""
        dicts = [
            ATOMIC_NUMBERS,
            ATOMIC_MASSES,
            ATOMIC_RADII,
            COVALENT_RADII,
            ELECTRONEGATIVITIES,
            IONIZATION_ENERGIES,
            ELECTRON_AFFINITIES,
            MELTING_POINTS,
            COMMON_OXIDATION_STATES,
            GROUP,
            PERIOD,
            THERMAL_CONDUCTIVITIES,
        ]
        for d in dicts:
            for key in d.keys():
                assert 1 <= len(key) <= 2, f"Invalid key length: '{key}'"
                assert key[0].isupper(), f"First letter not uppercase: '{key}'"
                if len(key) == 2:
                    assert key[1].islower(), f"Second letter not lowercase: '{key}'"

    def test_atomic_numbers_keys_are_valid_symbols(self):
        """All keys must be valid 1-2 letter element symbols."""
        for symbol in ATOMIC_NUMBERS.keys():
            assert 1 <= len(symbol) <= 2, f"Invalid symbol length: {symbol}"
            assert symbol[0].isupper(), f"First letter not uppercase: {symbol}"
            if len(symbol) == 2:
                assert symbol[1].islower(), f"Second letter not lowercase: {symbol}"

    def test_no_duplicate_symbols(self):
        """No duplicate element symbols in keys."""
        symbols = list(ATOMIC_NUMBERS.keys())
        assert len(symbols) == len(set(symbols)), "Duplicate symbols found"

    def test_atomic_numbers_are_positive(self):
        """All atomic numbers must be in range 1-118."""
        for symbol, z in ATOMIC_NUMBERS.items():
            assert 1 <= z <= 118, f"{symbol}: invalid Z={z}"

    def test_atomic_numbers_are_unique(self):
        """Each element must have a unique atomic number."""
        numbers = list(ATOMIC_NUMBERS.values())
        assert len(numbers) == len(set(numbers)), "Duplicate atomic numbers"

    def test_atomic_masses_positive(self):
        """All atomic masses must be positive."""
        for symbol, mass in ATOMIC_MASSES.items():
            assert mass > 0, f"{symbol}: non-positive mass={mass}"

    def test_electronegativity_range_valid(self):
        """Electronegativities should be in [0, 4.5] (Pauling scale)."""
        for symbol, en in ELECTRONEGATIVITIES.items():
            assert 0.0 <= en <= 4.5, f"{symbol}: EN={en} out of range"

    def test_ionization_energy_positive(self):
        """Ionization energies must be positive (eV)."""
        for symbol, ie in IONIZATION_ENERGIES.items():
            assert ie > 0, f"{symbol}: IE={ie} non-positive"

    def test_electron_affinity_realistic(self):
        """Electron affinities must be finite (NIST includes negative values for some elements)."""
        for symbol, ea in ELECTRON_AFFINITIES.items():
            assert math.isfinite(ea), f"{symbol}: EA={ea} not finite"

    def test_radii_non_negative(self):
        """Covalent and atomic radii must be >= 0."""
        for symbol in ATOMIC_RADII.keys():
            assert ATOMIC_RADII[symbol] >= 0, f"{symbol}: negative atomic radius"
        for symbol in COVALENT_RADII.keys():
            assert COVALENT_RADII[symbol] >= 0, f"{symbol}: negative covalent radius"

    def test_melting_points_non_negative(self):
        """Melting points must be >= 0 K."""
        for symbol, mp in MELTING_POINTS.items():
            assert mp >= 0, f"{symbol}: negative melting point={mp}"

    def test_oxidation_states_not_empty(self):
        """Every element in COMMON_OXIDATION_STATES must have at least one state."""
        for symbol, states in COMMON_OXIDATION_STATES.items():
            assert len(states) > 0, f"{symbol}: no oxidation states"

    def test_oxidation_states_are_integers(self):
        """All oxidation states must be integers."""
        for symbol, states in COMMON_OXIDATION_STATES.items():
            for state in states:
                assert isinstance(state, int), f"{symbol}: non-int state={state}"

    def test_group_period_in_valid_range(self):
        """Group must be 1-18, period must be 1-7."""
        common_keys = set(GROUP.keys()) & set(PERIOD.keys())
        for symbol in common_keys:
            assert 1 <= GROUP[symbol] <= 18, f"{symbol}: invalid group={GROUP[symbol]}"
            assert 1 <= PERIOD[symbol] <= 7, f"{symbol}: invalid period={PERIOD[symbol]}"

    def test_thermal_conductivity_non_negative(self):
        """Thermal conductivities must be >= 0."""
        for symbol, tc in THERMAL_CONDUCTIVITIES.items():
            assert tc >= 0, f"{symbol}: negative thermal conductivity={tc}"


class TestCommonElementData:
    """Smoke tests for well-known elements."""

    def test_hydrogen(self):
        assert ATOMIC_NUMBERS["H"] == 1
        assert ATOMIC_MASSES["H"] == pytest.approx(1.008, abs=0.001)
        assert ELECTRONEGATIVITIES["H"] == pytest.approx(2.20, abs=0.01)

    def test_carbon(self):
        assert ATOMIC_NUMBERS["C"] == 6
        assert ATOMIC_MASSES["C"] == pytest.approx(12.011, abs=0.001)
        assert ELECTRONEGATIVITIES["C"] == pytest.approx(2.55, abs=0.01)

    def test_oxygen(self):
        assert ATOMIC_NUMBERS["O"] == 8
        assert ELECTRONEGATIVITIES["O"] == pytest.approx(3.44, abs=0.01)

    def test_iron(self):
        assert ATOMIC_NUMBERS["Fe"] == 26

    def test_gold(self):
        assert ATOMIC_NUMBERS["Au"] == 79

    def test_uranium(self):
        assert ATOMIC_NUMBERS["U"] == 92


# ──────────────────────────────────────────────────────────────────────────
# 2. weighted_mean tests
# ──────────────────────────────────────────────────────────────────────────

class TestWeightedMean:
    """weighted_mean(elements, counts, prop_dict, default=0.0)"""

    def test_simple_mean(self):
        result = weighted_mean(["H", "H", "O"], [1, 1, 1], ATOMIC_MASSES)
        # H=1.008, O=15.999 => (1.008+1.008+15.999)/3
        expected = (1.008 + 1.008 + 15.999) / 3
        assert result == pytest.approx(expected, abs=0.01)

    def test_weighted_mean(self):
        result = weighted_mean(["H", "O"], [2, 1], ATOMIC_MASSES)
        # (2*1.008 + 1*15.999) / 3
        expected = (2 * 1.008 + 15.999) / 3
        assert result == pytest.approx(expected, abs=0.01)

    def test_single_element(self):
        result = weighted_mean(["Fe"], [1], ATOMIC_MASSES)
        assert result == pytest.approx(ATOMIC_MASSES["Fe"], abs=0.001)

    def test_empty_returns_default(self):
        assert weighted_mean([], [], ATOMIC_MASSES) == 0.0

    def test_zero_counts_returns_default(self):
        assert weighted_mean(["H", "O"], [0, 0], ATOMIC_MASSES) == 0.0

    def test_all_same_element(self):
        result = weighted_mean(["C", "C", "C"], [1, 2, 3], ATOMIC_MASSES)
        assert result == pytest.approx(ATOMIC_MASSES["C"], abs=0.001)

    def test_with_electronegativity(self):
        result = weighted_mean(["Na", "Cl"], [1, 1], ELECTRONEGATIVITIES)
        expected = (ELECTRONEGATIVITIES["Na"] + ELECTRONEGATIVITIES["Cl"]) / 2
        assert result == pytest.approx(expected, abs=0.01)

    def test_with_ionization_energy(self):
        result = weighted_mean(["H", "He"], [1, 1], IONIZATION_ENERGIES)
        expected = (IONIZATION_ENERGIES["H"] + IONIZATION_ENERGIES["He"]) / 2
        assert result == pytest.approx(expected, abs=0.01)

    def test_unknown_element_skipped_but_count_included(self):
        """Unknown elements are skipped but their count is still in total divisor."""
        result = weighted_mean(["H", "Xx"], [1, 1], ATOMIC_MASSES)
        # H mass = 1.008, total = 2, but Xx not found -> 1.008 / 2 = 0.504
        assert result == pytest.approx(0.504, abs=0.001)

    def test_all_unknown_returns_default(self):
        assert weighted_mean(["Xx", "Yy"], [1, 1], ATOMIC_MASSES) == 0.0

    def test_custom_default(self):
        assert weighted_mean([], [], ATOMIC_MASSES, default=-1.0) == -1.0


# ──────────────────────────────────────────────────────────────────────────
# 3. weighted_range tests
# ──────────────────────────────────────────────────────────────────────────

class TestWeightedRange:
    """weighted_range(elements, prop_dict, default=0.0) — NO counts parameter."""

    def test_basic_range(self):
        result = weighted_range(["H", "O"], ATOMIC_MASSES)
        expected = ATOMIC_MASSES["O"] - ATOMIC_MASSES["H"]
        assert result == pytest.approx(expected, abs=0.01)

    def test_single_element_returns_default(self):
        assert weighted_range(["Fe"], ATOMIC_MASSES) == 0.0

    def test_empty_returns_default(self):
        assert weighted_range([], ATOMIC_MASSES) == 0.0

    def test_multiple_elements_range(self):
        result = weighted_range(["H", "C", "O"], ATOMIC_MASSES)
        expected = ATOMIC_MASSES["O"] - ATOMIC_MASSES["H"]
        assert result == pytest.approx(expected, abs=0.01)

    def test_with_electronegativity(self):
        result = weighted_range(["Na", "Cl"], ELECTRONEGATIVITIES)
        expected = ELECTRONEGATIVITIES["Cl"] - ELECTRONEGATIVITIES["Na"]
        assert result == pytest.approx(expected, abs=0.01)

    def test_unknown_element_filtered(self):
        """Unknown elements are silently skipped."""
        result = weighted_range(["H", "Xx", "O"], ATOMIC_MASSES)
        expected = ATOMIC_MASSES["O"] - ATOMIC_MASSES["H"]
        assert result == pytest.approx(expected, abs=0.01)

    def test_all_unknown_returns_default(self):
        assert weighted_range(["Xx", "Yy"], ATOMIC_MASSES) == 0.0

    def test_custom_default(self):
        assert weighted_range(["Fe"], ATOMIC_MASSES, default=-1.0) == -1.0


# ──────────────────────────────────────────────────────────────────────────
# 4. weighted_std tests
# ──────────────────────────────────────────────────────────────────────────

class TestWeightedStd:
    """weighted_std(elements, counts, prop_dict, default=0.0)"""

    def test_zero_std(self):
        result = weighted_std(["Fe", "Fe", "Fe"], [1, 1, 1], ATOMIC_MASSES)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_basic_std(self):
        result = weighted_std(["H", "O"], [1, 1], ATOMIC_MASSES)
        # Population std of [1.008, 15.999] with equal weights
        mean = (1.008 + 15.999) / 2
        expected = np.sqrt(((1.008 - mean) ** 2 + (15.999 - mean) ** 2) / 2)
        assert result == pytest.approx(expected, abs=0.01)

    def test_single_element_returns_default(self):
        assert weighted_std(["Fe"], [1], ATOMIC_MASSES) == 0.0

    def test_empty_returns_default(self):
        assert weighted_std([], [], ATOMIC_MASSES) == 0.0

    def test_zero_counts_returns_default(self):
        assert weighted_std(["H", "O"], [0, 0], ATOMIC_MASSES) == 0.0

    def test_custom_default(self):
        assert weighted_std([], [], ATOMIC_MASSES, default=-1.0) == -1.0


# ──────────────────────────────────────────────────────────────────────────
# 5. assign_oxidation_states tests
# ──────────────────────────────────────────────────────────────────────────

class TestAssignOxidationStates:
    """assign_oxidation_states(elements, counts) assigns states using electroneutrality."""

    def test_single_element_returns_zero(self):
        """Elemental form: oxidation state is 0."""
        result = assign_oxidation_states(["Na"], [1])
        assert result == {"Na": 0}

    def test_iron_elemental(self):
        result = assign_oxidation_states(["Fe"], [1])
        assert result == {"Fe": 0}

    def test_oxygen_elemental(self):
        result = assign_oxidation_states(["O"], [2])
        assert result == {"O": 0}

    def test_nacl(self):
        result = assign_oxidation_states(["Na", "Cl"], [1, 1])
        assert result["Na"] == 1
        assert result["Cl"] == -1

    def test_mgo(self):
        result = assign_oxidation_states(["Mg", "O"], [1, 1])
        assert result["Mg"] == 2
        assert result["O"] == -2

    def test_water(self):
        result = assign_oxidation_states(["H", "O"], [2, 1])
        assert result["H"] == 1
        assert result["O"] == -2

    def test_naf(self):
        result = assign_oxidation_states(["Na", "F"], [1, 1])
        assert result["Na"] == 1
        assert result["F"] == -1

    def test_unknown_element_skipped(self):
        """Unknown elements get value via electroneutrality fallback."""
        result = assign_oxidation_states(["Na", "Xx"], [1, 1])
        assert "Na" in result

    def test_empty_returns_empty(self):
        result = assign_oxidation_states([], [])
        assert result == {}

    def test_returns_dict_with_int_values(self):
        result = assign_oxidation_states(["Li", "Fe", "P", "O"], [1, 1, 1, 4])
        for element, state in result.items():
            assert isinstance(state, int), f"{element}: {state} is not int"


# ──────────────────────────────────────────────────────────────────────────
# 6. count_oxidation_states tests
# ──────────────────────────────────────────────────────────────────────────

class TestCountOxidationStates:
    """count_oxidation_states(elements, counts) returns oxidation-state fractions."""

    def test_single_element_zero_state(self):
        """Elemental: 100% in state 0."""
        result = count_oxidation_states(["Na"], [1])
        assert result["ox_state_0_frac"] == 1.0
        assert result["ox_n_zero"] == 1.0
        assert result["ox_n_positive"] == 0.0
        assert result["ox_n_negative"] == 0.0

    def test_nacl(self):
        result = count_oxidation_states(["Na", "Cl"], [1, 1])
        assert result["ox_state_+1_frac"] == 0.5
        assert result["ox_state_-1_frac"] == 0.5
        assert result["ox_n_positive"] == 0.5
        assert result["ox_n_negative"] == 0.5

    def test_mgo(self):
        result = count_oxidation_states(["Mg", "O"], [1, 1])
        assert result["ox_state_+2_frac"] == 0.5
        assert result["ox_state_-2_frac"] == 0.5

    def test_water(self):
        result = count_oxidation_states(["H", "O"], [2, 1])
        # H(+1): 2/3, O(-2): 1/3
        assert result["ox_state_+1_frac"] == pytest.approx(2 / 3, abs=0.01)
        assert result["ox_state_-2_frac"] == pytest.approx(1 / 3, abs=0.01)

    def test_empty(self):
        result = count_oxidation_states([], [])
        assert result["ox_state_0_frac"] == 0.0
        assert result["ox_n_zero"] == 0.0

    def test_all_keys_present(self):
        """Result must have all expected keys."""
        result = count_oxidation_states(["Na", "Cl"], [1, 1])
        for ox in range(-3, 8):
            key = f'ox_state_{ox:+d}_frac' if ox != 0 else 'ox_state_0_frac'
            assert key in result
        assert "ox_n_positive" in result
        assert "ox_n_negative" in result
        assert "ox_n_zero" in result

    def test_fractions_sum_to_one(self):
        """All state fractions should sum to 1.0."""
        result = count_oxidation_states(["Na", "Cl"], [1, 1])
        total = sum(v for k, v in result.items() if k.startswith("ox_state_"))
        assert total == pytest.approx(1.0, abs=0.001)


# ──────────────────────────────────────────────────────────────────────────
# 7. Greenwood & Earnshaw validation
# ──────────────────────────────────────────────────────────────────────────

class TestGreenwoodEarnshawAssignments:
    """Validate key oxidation state data against known chemistry."""

    def test_iron_most_common_is_plus_three(self):
        states = COMMON_OXIDATION_STATES["Fe"]
        assert states[0] == 3, f"Expected Fe most common = +3, got {states[0]}"

    def test_copper_most_common_is_plus_two(self):
        states = COMMON_OXIDATION_STATES["Cu"]
        assert states[0] == 2, f"Expected Cu most common = +2, got {states[0]}"

    def test_manganese_most_common_is_plus_two(self):
        states = COMMON_OXIDATION_STATES["Mn"]
        assert states[0] == 2, f"Expected Mn most common = +2, got {states[0]}"

    def test_oxygen_always_negative_two(self):
        states = COMMON_OXIDATION_STATES["O"]
        assert states[0] == -2

    def test_fluorine_always_negative_one(self):
        states = COMMON_OXIDATION_STATES["F"]
        assert states[0] == -1
        assert len(states) == 1

    def test_alkali_metals_always_plus_one(self):
        for symbol in ["Li", "Na", "K", "Rb", "Cs"]:
            assert COMMON_OXIDATION_STATES[symbol][0] == 1

    def test_alkaline_earth_always_plus_two(self):
        for symbol in ["Be", "Mg", "Ca", "Sr", "Ba"]:
            assert COMMON_OXIDATION_STATES[symbol][0] == 2

    def test_transition_metals_have_multiple_states(self):
        for symbol in ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni"]:
            assert len(COMMON_OXIDATION_STATES[symbol]) >= 2, (
                f"{symbol} should have multiple oxidation states"
            )

    def test_lanthanides_primarily_plus_three(self):
        """Only test lanthanides that are actually present in the table."""
        lanthanides = [
            "La", "Ce", "Pr", "Nd",
        ]
        for symbol in lanthanides:
            if symbol in COMMON_OXIDATION_STATES:
                assert COMMON_OXIDATION_STATES[symbol][0] == 3, (
                    f"{symbol} expected primary state +3, got {COMMON_OXIDATION_STATES[symbol][0]}"
                )

    def test_actinides_primarily_positive(self):
        """Only test actinides that are actually present in the table."""
        actinides = [
            "Th", "U",
        ]
        for symbol in actinides:
            if symbol in COMMON_OXIDATION_STATES:
                primary = COMMON_OXIDATION_STATES[symbol][0]
                assert primary in [3, 4, 5, 6], (
                    f"{symbol} has unexpected primary state: {primary}"
                )
