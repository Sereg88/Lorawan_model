"""Юнит-тесты модулей MAC-уровня (src/mac.py) и радиоканала/ADR (src/channel_adr.py)."""

import pytest
from src.mac import calc_rx_window_duration_ms, calc_class_a_timings, calc_class_b_current_addon
from src.channel_adr import resolve_adr, fit_path_loss_params


def test_rx2_window_independent_of_tx_sf_hypothesis_g1() -> None:
    """
    Проверка свойства формулы (2.2) для гипотезы Г1:
    длительность окна RX2 не зависит от SF восходящего сообщения и равна N_sym * T_sym(SF12).
    """
    t_sf7 = calc_class_a_timings(sf_tx=7, payload_bytes=12, n_sym_rx=8)
    t_sf12 = calc_class_a_timings(sf_tx=12, payload_bytes=12, n_sym_rx=8)

    assert t_sf7.rx2_ms == pytest.approx(8 * 32.768, abs=1e-3)
    assert t_sf7.rx2_ms == pytest.approx(t_sf12.rx2_ms, abs=1e-6)
    assert t_sf7.rx1_ms == pytest.approx(8 * 1.024, abs=1e-3)


@pytest.mark.parametrize("sf", [7, 9, 12])
def test_rx_windows_open_at_receive_delays(sf) -> None:
    """RX1 открывается через RECEIVE_DELAY1 = 1 с, RX2 — через RECEIVE_DELAY2 = 2 с после конца передачи."""
    t = calc_class_a_timings(sf_tx=sf, payload_bytes=12, n_sym_rx=8)

    assert t.wait_rx1_ms == pytest.approx(1000.0)
    assert t.wait_rx1_ms + t.rx1_ms + t.wait_rx2_ms == pytest.approx(2000.0)


def test_class_b_ping_period() -> None:
    """Интервал между ping-слотами равен 0,96 * 2^k с (окно 122,88 с внутри периода маяка 128 с)."""
    assert calc_class_b_current_addon(4, 1.0, 0.1).ping_period_s == pytest.approx(15.36)
    assert calc_class_b_current_addon(1, 1.0, 0.1).ping_period_s == pytest.approx(1.92)


def test_adr_reduces_power_in_2_db_steps() -> None:
    """После выхода на SF7 оставшиеся шаги ADR снижают мощность по индексу TXPower (2 дБ за шаг)."""
    res = resolve_adr(distance_km=0.5, adr_enabled=True, tx_power_dbm=10.0, min_tx_power_dbm=2.0)
    extra_steps = res.n_steps - (12 - 7)

    assert res.final_sf == 7
    assert res.final_tx_power_dbm == pytest.approx(max(2.0, 10.0 - 2.0 * extra_steps))


def test_class_b_current_addon_hypothesis_g3() -> None:
    """
    Проверка формулы (2.5) для гипотезы Г3:
    с уменьшением периодичности k (7 -> 4 -> 1) число ping-слотов и средний ток растут.
    """
    b_k7 = calc_class_b_current_addon(periodicity_k=7, q_beacon_mc=2.0, q_ping_mc=0.4)
    b_k4 = calc_class_b_current_addon(periodicity_k=4, q_beacon_mc=2.0, q_ping_mc=0.4)
    b_k1 = calc_class_b_current_addon(periodicity_k=1, q_beacon_mc=2.0, q_ping_mc=0.4)

    assert b_k7.slots_per_beacon == 1
    assert b_k4.slots_per_beacon == 8
    assert b_k1.slots_per_beacon == 64
    assert b_k7.delta_i_class_b_ma < b_k4.delta_i_class_b_ma < b_k1.delta_i_class_b_ma


def test_adr_switches_to_low_sf_at_short_distance_hypothesis_g4() -> None:
    """Проверка формулы (2.7) для гипотезы Г4: на малом расстоянии ADR переводит узел с SF12 на SF7."""
    res_close = resolve_adr(distance_km=0.1, adr_enabled=True)
    res_far = resolve_adr(distance_km=8.0, adr_enabled=True)

    assert res_close.final_sf in (7, 8, 9)
    assert res_far.final_sf > res_close.final_sf


def test_fit_path_loss_params_series_c7() -> None:
    """Проверка восстановления параметров L0 и gamma по синтетическим точкам серии С7."""
    distances = [0.05, 0.5, 1.5, 3.0]
    # Истинные L0 = 72.0 дБ, gamma = 3.0 при P_tx = 10 дБм
    rssi = [10.0 - (72.0 + 30.0 * 0.0), 10.0 - (72.0 + 30.0 * 1.0), 10.0 - (72.0 + 30.0 * 1.47712), 10.0 - (72.0 + 30.0 * 1.77815)]
    l0, gamma = fit_path_loss_params(distances, rssi, tx_power_dbm=10.0, ref_distance_km=0.05)

    assert l0 == pytest.approx(72.0, abs=0.05)
    assert gamma == pytest.approx(3.0, abs=0.05)