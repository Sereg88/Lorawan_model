"""
Юнит-тесты модулей узла (src/node.py) и источника питания (src/battery.py).
Проверяет выполнение гипотез Г1, Г2, Г3, Г5 и Г6 на аналитической модели.
"""

from pathlib import Path
import pytest

from src.states import load_config, NodeState, PowerCircuit
from src.battery import calc_battery_lifetime_hours
from src.node import calc_sleep_breakeven_time_s, evaluate_node_energy_budget


@pytest.fixture(scope="module")
def configs():
    root = Path(__file__).resolve().parent.parent
    return {
        "apriori": load_config(root / "config" / "apriori_params.yaml"),
        "calibrated": load_config(root / "config" / "calibrated_params.yaml"),
    }


def test_hypothesis_g1_tx_energy_grows_less_than_toa(configs) -> None:
    """
    Гипотеза Г1: переход от SF7 к SF12 увеличивает ToA в ~24 раза, но энергия
    радиотранзакции возрастает меньше (~на порядок, в 6–12 раз) из-за фиксированного RX2 на DR0.
    """
    cfg = configs["apriori"]
    res_sf7 = evaluate_node_energy_budget(cfg, sf=7, payload_bytes=12)
    res_sf12 = evaluate_node_energy_budget(cfg, sf=12, payload_bytes=12)

    energy_ratio = res_sf12.e_tx_rx_cycle_mj / res_sf7.e_tx_rx_cycle_mj
    assert 5.0 < energy_ratio < 15.0
    # На SF7 доля окна RX2 в энергии транзакции значительно выше, чем на SF12
    assert res_sf7.share_rx2_in_tx_rx_pct > res_sf12.share_rx2_in_tx_rx_pct


def test_hypothesis_g2_sleep_dominates_at_long_periods(configs) -> None:
    """
    Гипотеза Г2: при больших периодах передачи (например, 6 часов = 21600 с)
    выбор режима сна влияет на срок службы сильнее, чем выбор SF.
    """
    cfg = configs["calibrated"]
    t_long = 6 * 3600.0  # 6 часов

    # Сравнение SF7 vs SF12 в глубоком сне
    r_sf7_deep = evaluate_node_energy_budget(cfg, sf=7, sleep_mode="deep_sleep", tx_period_s=t_long)
    r_sf12_deep = evaluate_node_energy_budget(cfg, sf=12, sleep_mode="deep_sleep", tx_period_s=t_long)

    # Переход в light_sleep при фиксированном SF7
    r_sf7_light = evaluate_node_energy_budget(cfg, sf=7, sleep_mode="light_sleep", tx_period_s=t_long)

    sf_impact_ratio = r_sf7_deep.battery.lifetime_years / r_sf12_deep.battery.lifetime_years
    sleep_impact_ratio = r_sf7_deep.battery.lifetime_years / r_sf7_light.battery.lifetime_years

    assert sleep_impact_ratio > sf_impact_ratio
    assert r_sf7_deep.share_sleep_pct > 50.0


def test_hypothesis_g5_active_wait_vs_light_sleep(configs) -> None:
    """
    Гипотеза Г5: активное ожидание окон приёма делает паузы WAIT_RX1/WAIT_RX2
    одной из основных составляющих бюджета и сильнее влияет на энергию, чем частота ЦП.
    """
    cfg = configs["apriori"]
    r_light_80 = evaluate_node_energy_budget(cfg, cpu_freq_mhz=80, wait_mode="light_sleep")
    r_light_240 = evaluate_node_energy_budget(cfg, cpu_freq_mhz=240, wait_mode="light_sleep")
    r_active_80 = evaluate_node_energy_budget(cfg, cpu_freq_mhz=80, wait_mode="active")

    diff_freq = abs(r_light_240.e_tx_rx_cycle_mj - r_light_80.e_tx_rx_cycle_mj)
    diff_wait = r_active_80.e_tx_rx_cycle_mj - r_light_80.e_tx_rx_cycle_mj

    assert diff_wait > diff_freq
    assert r_active_80.share_wait_windows_pct > r_light_80.share_wait_windows_pct


def test_hypothesis_g6_apriori_underestimates_sleep_current(configs) -> None:
    """
    Гипотеза Г6: априорная модель по даташитам занижает ток сна и средний ток
    по сравнению с калиброванной моделью (учитывающей ток стабилизаторов платы).
    """
    r_apr = evaluate_node_energy_budget(configs["apriori"], tx_period_s=600.0)
    r_cal = evaluate_node_energy_budget(configs["calibrated"], tx_period_s=600.0)

    assert r_cal.i_sleep_total_ua > r_apr.i_sleep_total_ua
    assert r_cal.i_avg_total_ma > r_apr.i_avg_total_ma


def test_wakeup_energy_included_in_wu_state(configs) -> None:
    """Состояние WU включает выход из сна: после глубокого сна (перезагрузка) заряд WU больше, чем после light sleep."""
    cfg = configs["apriori"]
    r_deep = evaluate_node_energy_budget(cfg, sleep_mode="deep_sleep")
    r_light = evaluate_node_energy_budget(cfg, sleep_mode="light_sleep")
    modes = cfg["mcu_configs"]["K1_ESP32"]["sleep_modes"]
    dq_expected = (modes["deep_sleep"]["wakeup_energy_uj"] - modes["light_sleep"]["wakeup_energy_uj"]) / 3.3 / 1000.0

    q_wu_deep = r_deep.profile.get_state_charge_mc(NodeState.WU, PowerCircuit.MCU_BOARD)
    q_wu_light = r_light.profile.get_state_charge_mc(NodeState.WU, PowerCircuit.MCU_BOARD)
    assert q_wu_deep - q_wu_light == pytest.approx(dq_expected, rel=1e-6)


def test_frequent_sensor_polling_requires_wakeups(configs) -> None:
    """
    При опросе датчика 1 раз в секунду и периоде передачи 60 с МК 59 раз дополнительно выходит из сна,
    и средний ток растёт не меньше чем на 59 * Q_wu / T_tx.
    """
    cfg = configs["apriori"]
    r_rare = evaluate_node_energy_budget(cfg, sleep_mode="deep_sleep", sensor_freq_hz=1.0 / 60.0, tx_period_s=60.0)
    r_often = evaluate_node_energy_budget(cfg, sleep_mode="deep_sleep", sensor_freq_hz=1.0, tx_period_s=60.0)
    q_wake_mc = cfg["mcu_configs"]["K1_ESP32"]["sleep_modes"]["deep_sleep"]["wakeup_energy_uj"] / 3.3 / 1000.0

    assert r_often.i_avg_total_ma - r_rare.i_avg_total_ma >= 59.0 * q_wake_mc / 60.0


def test_atmega_power_down_allows_timer_wakeup(configs) -> None:
    """Ток power-down ATmega328P задан с включённым WDT (единицы мкА), иначе периодическое пробуждение невозможно."""
    i_pd_ua = 1000.0 * configs["apriori"]["mcu_configs"]["K2_ATMEGA"]["sleep_modes"]["power_down"]["current_ma"]
    assert i_pd_ua >= 3.0


def test_sleep_breakeven_time_formula_2_1() -> None:
    """Проверка формулы (2.1): время безубыточности перехода из light_sleep в deep_sleep для ESP32."""
    # light_sleep: 800 мкА, 8 мкДж; deep_sleep: 10 мкА, 1250 мкДж; U = 3.3 В
    t_be_s = calc_sleep_breakeven_time_s(
        e_wu_shallow_uj=8.0,
        e_wu_deep_uj=1250.0,
        i_sleep_shallow_ua=800.0,
        i_sleep_deep_ua=10.0,
        voltage_v=3.3,
    )
    assert t_be_s == pytest.approx((1250.0 - 8.0) / (3.3 * 790.0), rel=1e-4)


def test_battery_self_discharge_reduces_lifetime_formula_2_8() -> None:
    """Проверка формулы (2.8): учёт саморазряда снижает расчётный срок службы."""
    t_ideal = calc_battery_lifetime_hours(avg_current_ma=0.05, self_discharge_lambda_per_hour=0.0)
    t_li_socl2 = calc_battery_lifetime_hours(avg_current_ma=0.05, self_discharge_lambda_per_hour=1.14e-6)
    t_li_ion = calc_battery_lifetime_hours(avg_current_ma=0.05, self_discharge_lambda_per_hour=4.11e-5)

    assert t_ideal > t_li_socl2 > t_li_ion