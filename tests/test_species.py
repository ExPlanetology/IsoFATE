from scipy import constants as scipy_constants

from isofate.constants import const
from isofate.species import (
    DEFAULT_BINARY_DIFFUSION,
    DEFAULT_SPECIES,
    SYMBOLS,
    BinaryDiffusionCoefficients,
    IsoFATESpecies,
)

EXPECTED_ORDER = ("H", "He", "D", "O", "C", "N", "S")


def test_symbols_in_expected_order():
    assert SYMBOLS == EXPECTED_ORDER


def test_default_species_matches_symbols():
    assert DEFAULT_SPECIES.species == SYMBOLS


def test_mass_by_symbol_matches_atomic_masses():
    assert tuple(DEFAULT_SPECIES.mass_by_symbol[symbol] for symbol in SYMBOLS) == tuple(
        DEFAULT_SPECIES.atomic_masses.tolist()
    )


def test_custom_species_subset():
    species = IsoFATESpecies(("H", "O"))
    assert species.species == ("H", "O")
    assert species.atomic_masses[0] == DEFAULT_SPECIES.mass_by_symbol["H"]
    assert species.atomic_masses[1] == DEFAULT_SPECIES.mass_by_symbol["O"]


def test_binary_diffusion_symmetric():
    T = 500.0
    for species1, species2, prefactor, exponent, _source in BinaryDiffusionCoefficients.COEFFICIENTS:
        expected = prefactor * T**exponent
        assert DEFAULT_BINARY_DIFFUSION.get(species1, species2, T) == expected
        assert DEFAULT_BINARY_DIFFUSION.get(species2, species1, T) == expected


def test_binary_diffusion_falls_back_to_default_pair():
    T = 500.0
    h_he = DEFAULT_BINARY_DIFFUSION.get("H", "He", T)
    assert DEFAULT_BINARY_DIFFUSION.get("D", "O", T) == h_he


def test_binary_diffusion_subclass_overrides_coefficients():
    class OtherBinaryDiffusion(BinaryDiffusionCoefficients):
        COEFFICIENTS = (("H", "He", 1.0, 1.0, "test override"),)

    other = OtherBinaryDiffusion()
    assert other.get("H", "He", 10.0) == 10.0
    # DEFAULT_BINARY_DIFFUSION is unaffected by the subclass's override
    assert DEFAULT_BINARY_DIFFUSION.get("H", "He", 10.0) != 10.0


def test_scale_heights_ordered_per_species():
    T, g = 500.0, 20.0
    H = DEFAULT_SPECIES.scale_heights(T, g)
    for symbol, height in zip(SYMBOLS, H):
        expected = scipy_constants.Boltzmann * T / (DEFAULT_SPECIES.mass_by_symbol[symbol] * g)
        assert height == expected


def test_scale_heights_matches_ideal_gas_law_with_molar_mass():
    # H = kB*T/(m*g) (per-particle) should agree with H = R*T/(M*g) (per-mole), since
    # R = kB*N_A and M = m*N_A.
    T, g = 500.0, 20.0
    H = DEFAULT_SPECIES.scale_heights(T, g)
    molar_masses = {
        "H": const.M_H,
        "He": const.M_He,
        "D": const.M_D,
        "O": const.M_O,
        "C": const.M_C,
        "N": const.M_N,
        "S": const.M_S,
    }
    for symbol, height in zip(SYMBOLS, H):
        expected = const.R_gas * T / (molar_masses[symbol] * g)
        assert abs(height - expected) / expected < 1e-3
