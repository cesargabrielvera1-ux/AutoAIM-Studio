## AutoAIM Studio v1.4.0 — Crystal Structure Features (CIF Loading)

> **Note**: This table lists all 149 features extracted when loading CIF files directly via "Load Crystal Structures". Features are organized by category. 

| Category | Feature Name | Definition |
|----------|-------------|------------|
| **Lattice** | `lattice_a` | Lattice parameter a (Å) |
| **Lattice** | `lattice_b` | Lattice parameter b (Å) |
| **Lattice** | `lattice_c` | Lattice parameter c (Å) |
| **Lattice** | `lattice_alpha` | Lattice angle α (°) |
| **Lattice** | `lattice_beta` | Lattice angle β (°) |
| **Lattice** | `lattice_gamma` | Lattice angle γ (°) |
| **Lattice** | `lattice_volume` | Unit cell volume V (Å³) |
| **Lattice** | `lattice_abc_ratio` | Mean lattice parameter over maximum: [(a+b+c)/3] / max(a,b,c) |
| **Lattice** | `lattice_is_cubic` | Binary flag (1.0 if cubic within tolerance, else 0.0) |
| **Lattice** | `lattice_is_hexagonal` | Binary flag (1.0 if hexagonal within tolerance, else 0.0) |
| **Lattice** | `lattice_anisotropy` | Lattice shape anisotropy: std(a,b,c) / mean(a,b,c) |
| | | |
| **Composition** | `n_sites` | Number of atomic sites in the unit cell |
| **Composition** | `n_elements` | Number of unique chemical elements |
| **Composition** | `avg_atomic_number` | Weighted mean atomic number: Σ(Z_i × f_i) |
| **Composition** | `avg_atomic_mass` | Weighted mean atomic mass: Σ(mass_i × f_i) |
| **Composition** | `avg_electronegativity` | Weighted mean Pauling electronegativity: Σ(X_i × f_i) |
| **Composition** | `avg_ionization_energy` | Weighted mean ionization energy: Σ(IE_i × f_i) |
| **Composition** | `avg_electron_affinity` | Weighted mean electron affinity: Σ(EA_i × f_i) |
| **Composition** | `avg_covalent_radius` | Weighted mean covalent radius: Σ(r_cov_i × f_i) |
| **Composition** | `avg_atomic_radius` | Weighted mean atomic radius: Σ(r_atom_i × f_i) |
| **Composition** | `avg_density` | Weighted mean elemental density: Σ(ρ_i × f_i) |
| **Composition** | `electronegativity_variance` | Variance of Pauling electronegativity across elements (unweighted) |
| **Composition** | `radius_variance` | Variance of atomic radii across elements (unweighted) |
| **Composition** | `mass_variance` | Variance of atomic masses across elements (unweighted) |
| | | |
| **Structural** | `volume_per_atom` | Unit cell volume per atom: V_cell / n_sites (Å³/atom) |
| **Structural** | `packing_fraction` | Estimated packing fraction: Σ(V_sphere_i) / V_cell using covalent radii |
| **Structural** | `spacegroup_number` | Space group number (1–230) from SpacegroupAnalyzer |
| | | |
| **Derived** | `density` | Crystal density (g/cm³) from pymatgen Structure |
| **Derived** | `total_electrons` | Total number of electrons in the unit cell composition |
| | | |
| **Advanced Structural** | `mean_bond_length` | Mean nearest-neighbor distance (Å) from CrystalNN or distance cutoff |
| **Advanced Structural** | `std_bond_length` | Standard deviation of bond lengths (Å) |
| **Advanced Structural** | `min_bond_length` | Minimum nearest-neighbor distance (Å) |
| **Advanced Structural** | `max_bond_length` | Maximum nearest-neighbor distance (Å) |
| **Advanced Structural** | `mean_coordination_number` | Mean coordination number from CrystalNN or distance-based neighbors |
| **Advanced Structural** | `max_coordination_number` | Maximum coordination number observed in the structure |
| **Advanced Structural** | `n_bonds_per_atom` | Total bonds counted divided by number of atoms |
| **Advanced Structural** | `rdf_peak_1` | Position of 1st peak in radial distribution function (Å) |
| **Advanced Structural** | `rdf_peak_2` | Position of 2nd peak in radial distribution function (Å) |
| **Advanced Structural** | `rdf_peak_3` | Position of 3rd peak in radial distribution function (Å) |
| **Advanced Structural** | `rdf_peak_height_1` | Height (count) of 1st RDF peak |
| **Advanced Structural** | `rdf_peak_height_2` | Height (count) of 2nd RDF peak |
| **Advanced Structural** | `structure_complexity_index` | Complexity proxy: n_elements × ln(1 + n_sites) |
| **Advanced Structural** | `lattice_anisotropy` | Lattice shape anisotropy: std(a,b,c) / mean(a,b,c) |
| **Advanced Structural** | `surface_area_estimate` | Estimated surface area from lattice parameters: 2(ab + bc + ac) (Å²) |
| **Magpie Composition (106 features)** | `magpie_001` ... `magpie_106` | Composition descriptors (mean, std, min, max, range): AtomicNumber, MeltingT, 
Column, Row, NsValence, NpValence, NdValence, NfValence, NValence, NsUnfilled, NpUnfilled, 
NdUnfilled, NfUnfilled, NUnfilled, GSbandgap, GSmagmom, SpaceGroupNumber, etc. (106 total) |

**Total Features: 150**

---

## Feature Categories Summary

| Category | Count | Description |
|----------|-------|-------------|
| Lattice | 11 | Unit cell parameters, angles, volume, symmetry flags |
| Composition | 13 | Elemental statistics, weighted averages, variances |
| Structural | 3 | Per-atom volume, packing fraction, space group |
| Derived | 2 | Calculated properties (density, electron count) |
| Advanced Structural | 15 | Bond lengths, coordination numbers, RDF peaks, complexity |
| Magpie Composition | 106 | 106 elemental property statistics from Magpie |
| **TOTAL** | **150** | |
