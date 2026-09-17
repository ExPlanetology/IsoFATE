from isofate.species import DEFAULT_SPECIES, SYMBOLS, IsoFATESpecies

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
