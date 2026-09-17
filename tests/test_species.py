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


def test_species_default_binary_diffusion_is_shared_default():
    assert DEFAULT_SPECIES.binary_diffusion is DEFAULT_BINARY_DIFFUSION


def test_species_can_take_custom_binary_diffusion():
    class OtherBinaryDiffusion(BinaryDiffusionCoefficients):
        COEFFICIENTS = (("H", "He", 1.0, 1.0, "test override"),)

    species = IsoFATESpecies(binary_diffusion=OtherBinaryDiffusion())
    assert species.binary_diffusion.get("H", "He", 10.0) == 10.0
