"""
Юнит-тесты модуля физического уровня src/phy.py и загрузчика конфигураций src/states.py.
Проверяет точное совпадение расчётного времени передачи с Таблицей 2.4 отчёта.
"""

from pathlib import Path
import pytest

from src.phy import calc_symbol_duration_ms, calc_toa_ms, generate_table_2_4
from src.states import load_config


# Эталонные значения из Таблицы 2.4 отчёта (BW = 125 кГц, CR = 4/5, N_pre = 8, L_hdr = 13 байт)
EXPECTED_SYM_DURATION_MS = {
    7: 1.024,
    8: 2.048,
    9: 4.096,
    10: 8.192,
    11: 16.384,
    12: 32.768,
}

EXPECTED_TABLE_2_4_MS = {
    7:  {4: 51.5,   12: 61.7,   24: 82.2,   51: 118.0},
    8:  {4: 92.7,   12: 113.2,  24: 143.9,  51: 215.6},
    9:  {4: 164.9,  12: 205.8,  24: 267.3,  51: 390.1},
    10: {4: 329.7,  12: 411.6,  24: 493.6,  51: 698.4},
    11: {4: 659.5,  12: 823.3,  24: 1069.1, 51: 1560.6},
    12: {4: 1318.9, 12: 1482.8, 24: 1974.3, 51: 2793.5},
}

EXPECTED_SF12_SF7_RATIO = {
    4: 25.6,
    12: 24.0,
    24: 24.0,
    51: 23.7,
}


@pytest.mark.parametrize("sf,expected_t_sym", EXPECTED_SYM_DURATION_MS.items())
def test_symbol_duration_ms(sf: int, expected_t_sym: float) -> None:
    """Проверка длительности символа T_sym для SF7..SF12 при полосе 125 кГц."""
    assert calc_symbol_duration_ms(sf, bw_khz=125.0) == pytest.approx(expected_t_sym, abs=1e-4)


@pytest.mark.parametrize("sf", [7, 8, 9, 10, 11, 12])
@pytest.mark.parametrize("payload_bytes", [4, 12, 24, 51])
def test_table_2_4_toa_values(sf: int, payload_bytes: int) -> None:
    """Сверка всех 24 ячеек Таблицы 2.4 с точностью до 0.1 мс."""
    toa = calc_toa_ms(sf=sf, payload_bytes=payload_bytes)
    assert round(toa, 1) == pytest.approx(EXPECTED_TABLE_2_4_MS[sf][payload_bytes], abs=0.1)


@pytest.mark.parametrize("payload_bytes,expected_ratio", EXPECTED_SF12_SF7_RATIO.items())
def test_sf12_to_sf7_ratio_hypothesis_g1(payload_bytes: int, expected_ratio: float) -> None:
    """Проверка отношения времени передачи SF12/SF7 (нижняя строка Таблицы 2.4, гипотеза Г1)."""
    table = generate_table_2_4()
    ratio = table[12][payload_bytes] / table[7][payload_bytes]
    assert round(ratio, 1) == pytest.approx(expected_ratio, abs=0.1)


def test_config_files_load() -> None:
    """Проверка корректной загрузки обоих конфигурационных файлов YAML."""
    root = Path(__file__).resolve().parent.parent
    apriori = load_config(root / "config" / "apriori_params.yaml")
    calibrated = load_config(root / "config" / "calibrated_params.yaml")

    assert apriori["model_type"] == "apriori"
    assert calibrated["model_type"] == "calibrated"
    assert apriori["system"]["supply_voltage_v"] == 3.3
    assert calibrated["mcu_configs"]["K1_ESP32"]["board_quiescent_current_ma"] > 0.0