from isofate.constants import const
from isofate.interfaces import ATOMIC_MASSES, ELEMENTS, MASS_BY_SYMBOL, SYMBOLS

EXPECTED_ORDER = ("H", "He", "D", "O", "C", "N", "S")


def test_symbols_in_expected_order():
    assert SYMBOLS == EXPECTED_ORDER


def test_elements_atomic_masses_and_symbols_agree():
    assert tuple(e.symbol for e in ELEMENTS) == SYMBOLS
    assert tuple(e.mass for e in ELEMENTS) == ATOMIC_MASSES


def test_mass_by_symbol_matches_atomic_masses():
    assert tuple(MASS_BY_SYMBOL[symbol] for symbol in SYMBOLS) == ATOMIC_MASSES


def test_masses_match_physical_constants():
    expected = {
        "H": const.mu_H,
        "He": const.mu_He,
        "D": const.mu_D,
        "O": const.mu_O,
        "C": const.mu_C,
        "N": const.mu_N,
        "S": const.mu_S,
    }
    for symbol, mass in expected.items():
        assert MASS_BY_SYMBOL[symbol] == mass
