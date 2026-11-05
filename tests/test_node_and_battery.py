"""
Юнит-тесты модулей узла (src/node.py) и источника питания (src/battery.py).
Проверяет выполнение гипотез Г1, Г2, Г3, Г5 и Г6 на аналитической модели.
"""

from pathlib import Path
import pytest

from src.states import (
    NodeState,
    PowerCircuit,
    get_supported_band_plans,
    load_config,
    resolve_band_plan,
)
from src.battery import calc_battery_lifetime_hours
from src.node import calc_sleep_breakeven_time_s, evaluate_node_energy_budget


@pytest.mark.parametrize("model", ["apriori", "calibrated"])
def test_radio_descriptions_are_optional(configs, model):
    """Удаление пояснений не меняет расчёт и не отключает запрет несовместимого модуля."""
    from copy import deepcopy
    from src.radio import validate_radio_band_plan

    cfg = deepcopy(configs[model])
    for radio in cfg["radio_modules"].values():
        for key in ("compatibility_note", "tx_current_note", "sensitivity_conditions", "datasheet", "model_name", "chipset"):
            radio.pop(key, None)
    for module, plan in (("DXLR03900", "EU868"), ("DRF1278F", "EU433")):
        expected = evaluate_node_energy_budget(configs[model], radio_module=module, band_plan=plan)
        result = evaluate_node_energy_budget(cfg, radio_module=module, band_plan=plan)
        assert result.i_avg_total_ma == pytest.approx(expected.i_avg_total_ma)
        assert result.e_cycle_total_mj == pytest.approx(expected.e_cycle_total_mj)
    with pytest.raises(ValueError, match="LoRaWAN"):
        evaluate_node_energy_budget(cfg, radio_module="EBYTE", band_plan="EU433")
    with pytest.raises(ValueError, match="LoRaWAN"):
        validate_radio_band_plan(cfg["radio_modules"]["EBYTE"], cfg["band_plans"]["EU433"])


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


def test_radio_modules_are_available_in_both_models(configs) -> None:
    """Все радиомодули, доступные в интерфейсе, должны быть описаны в обеих конфигурациях модели."""
    radio_sets = {name: set(cfg["radio_modules"].keys()) for name, cfg in configs.items()}
    common_radios = set.intersection(*radio_sets.values())

    assert common_radios == radio_sets["apriori"] == radio_sets["calibrated"]


def test_band_plans_are_available_in_both_models(configs) -> None:
    """Все band plan'ы должны быть описаны и согласованы в обеих конфигурациях модели."""
    band_plan_sets = {name: set(cfg["band_plans"].keys()) for name, cfg in configs.items()}
    common_plans = set.intersection(*band_plan_sets.values())

    assert common_plans == band_plan_sets["apriori"] == band_plan_sets["calibrated"]


def test_module_band_plan_compatibility_is_configured(configs) -> None:
    """Список совместимых band plan'ов у модулей должен совпадать в априорной и откалиброванной моделях."""
    for radio_name in configs["apriori"]["radio_modules"]:
        apriori_plans = get_supported_band_plans(configs["apriori"], radio_name)
        calibrated_plans = get_supported_band_plans(configs["calibrated"], radio_name)
        assert apriori_plans == calibrated_plans


def test_dxlr03900_matches_datasheet_currents(configs) -> None:
    """DX-LR03-900T30D должен использовать паспортные токи из Table 4 технического руководства."""
    for cfg in configs.values():
        radio = cfg["radio_modules"]["DXLR03900"]
        assert radio["supported_band_plans"] == ["EU868", "AS923"]
        assert radio["sleep_current_ma"] == pytest.approx(2.61)
        assert radio["idle_current_ma"] == pytest.approx(6.75)
        assert radio["standby_current_ma"] == pytest.approx(22.28)
        assert radio["rx_current_ma"] == pytest.approx(26.42)
        assert 0 not in radio["tx_current_by_p_dbm"]  # Не выдаём оценку за вторую паспортную точку
        assert radio["tx_current_by_p_dbm"][28] == pytest.approx(179.0)
        assert radio["sensitivity_dbm"] == pytest.approx(-125.0)
        assert radio["max_tx_power_dbm"] == pytest.approx(28.0)


def test_band_plan_resolution_uses_selected_profile(configs) -> None:
    """Выбранный band plan должен подменять сетевые параметры независимо от радиомодуля."""
    plan_name, plan_cfg = resolve_band_plan(configs["apriori"], "AS923")

    assert plan_name == "AS923"
    assert plan_cfg["system"]["frequency_hz"] == 923200000
    assert plan_cfg["system"]["rx2_sf"] == 10


def test_incompatible_module_and_band_plan_raise_error(configs) -> None:
    """Нельзя рассчитывать модуль на band plan'е, которого нет в списке совместимости."""
    with pytest.raises(ValueError, match="не поддерживается радиомодулем"):
        evaluate_node_energy_budget(
            configs["apriori"],
            radio_module="DRF1278F",
            band_plan="EU868",
        )


def test_dxlr03900_can_run_with_eu868_band_plan(configs) -> None:
    """DXLR03900 должен корректно рассчитываться на совместимом band plan'е EU868."""
    result = evaluate_node_energy_budget(
        configs["apriori"],
        radio_module="DXLR03900",
        band_plan="EU868",
    )

    assert result.radio_module == "DXLR03900"
    assert result.band_plan == "EU868"
    assert result.tx_power_dbm >= 0.0


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


@pytest.mark.parametrize("model", ["apriori", "calibrated"])
def test_drf1278f_module_specs_not_chip_defaults(configs, model):
    radio = configs[model]["radio_modules"]["DRF1278F"]
    assert radio["frequency_range_mhz"] == [410, 460]
    assert radio["rx_current_ma"] == 12.0
    assert radio["sleep_current_ma"] == 0.001
    assert radio["sleep_current_kind"] == "maximum"
    assert radio["sensitivity_dbm"] == -136
    assert radio["max_tx_power_dbm"] == 20


def test_e220_stock_firmware_is_not_eu433(configs):
    from src.radio import validate_radio_band_plan
    radio = configs["apriori"]["radio_modules"]["EBYTE"]
    assert radio["model_name"] == "E220-400T22D"
    assert radio["chipset"] == "LLCC68"
    assert radio["frequency_range_mhz"] == [410.125, 493.125]
    assert radio["supported_band_plans"] == []
    assert radio["rx_current_ma"] == 16.8
    assert radio["sleep_current_ma"] == 0.005
    assert radio["sensitivity_dbm"] == -127
    assert radio["sensitivity_reference_sf"] is None
    assert radio["sf_by_bandwidth_khz"][125] == [7, 8, 9]
    with pytest.raises(ValueError, match="E220-400T22D"):
        evaluate_node_energy_budget(configs["apriori"], radio_module="EBYTE", band_plan="EU433")
    # Даже смена статуса прошивки не исправляет аппаратную сетку каналов.
    from copy import deepcopy
    modified = deepcopy(radio)
    modified["lorawan_support"] = "external_stack"
    with pytest.raises(ValueError, match="сетке каналов"):
        validate_radio_band_plan(modified, configs["apriori"]["band_plans"]["EU433"])


def test_module_sensitivity_anchor_and_estimated_sf_scaling(configs):
    from src.radio import module_sensitivity_dbm
    dx = configs["apriori"]["radio_modules"]["DXLR03900"]
    assert module_sensitivity_dbm(dx, 7) == -125
    assert module_sensitivity_dbm(dx, 12) == -137.5
    assert module_sensitivity_dbm(dx, 7, 250) == pytest.approx(-125 + 3.0103, abs=1e-4)
    drf = configs["apriori"]["radio_modules"]["DRF1278F"]
    assert module_sensitivity_dbm(drf, 12) == -136
    assert module_sensitivity_dbm(drf, 7) == -123.5


def test_receiver_sensitivity_affects_downlink_not_gateway_adr(configs):
    kwargs = dict(adr_enabled=True, distance_km=10)
    good = evaluate_node_energy_budget(configs["apriori"], **kwargs, receiver_sensitivity_dbm=-136)
    poor = evaluate_node_energy_budget(configs["apriori"], **kwargs, receiver_sensitivity_dbm=-90)
    assert good.adr_info == poor.adr_info
    assert good.sf == poor.sf
    assert good.downlink_feasible
    assert not poor.downlink_feasible
    assert not poor.link_feasible
    assert good.downlink_rx2_margin_db - poor.downlink_rx2_margin_db == pytest.approx(46)


def test_gateway_sensitivity_changes_adr_and_uplink(configs):
    kwargs = dict(distance_km=12, adr_enabled=True)
    good = evaluate_node_energy_budget(configs["apriori"], **kwargs, gateway_sensitivity_dbm=-145)
    poor = evaluate_node_energy_budget(configs["apriori"], **kwargs, gateway_sensitivity_dbm=-110)
    assert good.sf < poor.sf
    assert good.adr_info.link_feasible
    assert not poor.adr_info.link_feasible


def test_dxlr03900_two_rails_energy_and_reference_current(configs):
    from copy import deepcopy
    cfg = deepcopy(configs["apriori"])
    result = evaluate_node_energy_budget(cfg, radio_module="DXLR03900", band_plan="EU868")
    tx = result.profile.states[NodeState.TX][PowerCircuit.RADIO]
    assert tx.voltage_v == 5
    assert tx.current_ma == 179
    assert tx.energy_mj == pytest.approx(5 * 179 * tx.duration_s)
    assert result.profile.states[NodeState.TX][PowerCircuit.MCU_BOARD].voltage_v == 3.3
    active_energy = sum(result.profile.get_state_energy_mj(st) for st in (
        NodeState.WU, NodeState.TX, NodeState.WAIT_RX1, NodeState.RX1,
        NodeState.WAIT_RX2, NodeState.RX2, NodeState.PP))
    assert result.e_tx_rx_cycle_mj == pytest.approx(active_energy)
    assert result.e_cycle_total_mj == pytest.approx(result.i_avg_total_ma * 3.3 * result.tx_period_s)
    assert result.share_radio_pct + result.share_mcu_board_pct == pytest.approx(100)
    cfg["radio_modules"]["DXLR03900"]["power_conversion_efficiency"] = 0.8
    lossy = evaluate_node_energy_budget(cfg, radio_module="DXLR03900", band_plan="EU868")
    assert lossy.i_avg_total_ma > result.i_avg_total_ma
    assert lossy.battery.lifetime_years < result.battery.lifetime_years


def test_radio_sleep_current_is_not_lost_during_sensor_reads(configs):
    from copy import deepcopy
    cfg = deepcopy(configs["apriori"])
    # Равный ток во всех фазах: радиоцепь должна потреблять I*U*T независимо от частоты опроса.
    radio = cfg["radio_modules"]["DXLR03900"]
    for key in ("sleep_current_ma", "standby_current_ma", "rx_current_ma"):
        radio[key] = 10
    radio["tx_current_by_p_dbm"] = {28: 10}
    for sensor_freq in (1/60, 1):
        r = evaluate_node_energy_budget(cfg, radio_module="DXLR03900", band_plan="EU868", sensor_freq_hz=sensor_freq)
        assert r.share_radio_pct / 100 * r.e_cycle_total_mj == pytest.approx(5 * 10 * r.tx_period_s)


def test_regional_rx2_beacons_and_power_caps(configs):
    from copy import deepcopy
    cfg = deepcopy(configs["apriori"])
    expectations = {"EU433": (434665000, 9), "EU868": (869525000, 9), "AS923": (923200000, 9)}
    for plan, (rx2, beacon_sf) in expectations.items():
        p = cfg["band_plans"][plan]
        assert p["system"]["rx2_frequency_hz"] == rx2
        assert p["class_b"]["beacon_sf"] == beacon_sf
    assert cfg["band_plans"]["AS923"]["class_b"]["beacon_frequency_hz"] == 923400000
    cfg["band_plans"]["EU868"]["channel_adr"]["tx_power_dbm"] = 28
    r = evaluate_node_energy_budget(cfg, radio_module="DXLR03900", band_plan="EU868")
    assert r.tx_power_dbm == 16  # Возможности PA не отменяют предел плана


def test_nonfinite_sensitivity_is_rejected(configs):
    with pytest.raises(ValueError, match="чувствительности"):
        evaluate_node_energy_budget(configs["apriori"], receiver_sensitivity_dbm=float("nan"))



def test_class_b_requires_beacon_and_ping_link(configs):
    # При SF12 RX2 ещё принимается, но маяк SF9 уже нет.
    r = evaluate_node_energy_budget(
        configs["apriori"], device_class="B", sf=12, distance_km=10,
        receiver_sensitivity_dbm=-124,
    )
    assert r.downlink_feasible
    assert r.beacon_margin_db < 0
    assert not r.class_b_link_feasible
    assert not r.link_feasible


def test_frequency_dependence_of_path_loss(configs):
    from copy import deepcopy
    cfg = deepcopy(configs["apriori"])
    a = evaluate_node_energy_budget(cfg)
    p = cfg["band_plans"]["EU433"]["channel_adr"]
    # При той же частоте L0, определённый на вдвое меньшей частоте,
    # должен увеличиваться на 6.02 дБ.
    p["path_loss_reference_frequency_hz"] /= 2
    b = evaluate_node_energy_budget(cfg)
    assert a.adr_info.rssi_dbm - b.adr_info.rssi_dbm == pytest.approx(6.0206, abs=1e-4)


def test_tx_current_respects_hardware_limits(configs):
    from src.node import get_radio_tx_current_ma
    dx = configs["apriori"]["radio_modules"]["DXLR03900"]
    assert get_radio_tx_current_ma(dx, 10) == 179
    with pytest.raises(ValueError, match="Мощность"):
        get_radio_tx_current_ma(dx, 30)



@pytest.mark.parametrize("model", ["apriori", "calibrated"])
@pytest.mark.parametrize("profile", ["legacy", "v1_7"])
def test_p2p_bridge_energy_boundaries_and_duration(configs, model, profile):
    from src.p2p import evaluate_p2p_bridge, ENVELOPE_BYTES
    r = evaluate_p2p_bridge(configs[model], parameter_profile=profile, measured_airtime_ms=150)
    assert r.frequency_hz == 433125000
    assert r.forwarded_bytes == r.payload_bytes + ENVELOPE_BYTES
    assert r.forward_fport == 20
    assert r.airtime_source == "measured"
    assert sum(p.duration_ms for p in r.sensor_phases) == pytest.approx(r.tx_period_s * 1000)
    assert not {"RX1", "RX2", "WAIT_RX1", "WAIT_RX2"} & {p.name for p in r.sensor_phases}
    assert r.sensor_cycle_energy_mj == pytest.approx(sum(p.energy_mj for p in r.sensor_phases))
    assert r.sensor_avg_current_ma * r.reference_voltage_v * r.tx_period_s == pytest.approx(r.sensor_cycle_energy_mj)
    assert r.bridge_cycle_energy_mj == pytest.approx(
        r.bridge_lorawan.e_cycle_total_mj + r.bridge_p2p_rx_energy_mj + r.bridge_processing_energy_mj)
    assert r.bridge_avg_current_ma * r.bridge_lorawan.profile.voltage_v * r.tx_period_s == pytest.approx(r.bridge_cycle_energy_mj)
    assert r.bridge_lorawan.payload_bytes == r.forwarded_bytes
    assert r.bridge_lorawan.sensor_freq_hz == 0


def test_p2p_container_validates_id_length_and_crc():
    from src.p2p import encode_p2p_message, decode_p2p_message, prepare_lorawan_uplink, ENVELOPE_BYTES
    raw = b"temperature=23"
    packet = encode_p2p_message(65535, 0xFFFFFFFF, raw)
    assert len(packet) == len(raw) + ENVELOPE_BYTES
    msg = decode_p2p_message(packet)
    assert (msg.sensor_id, msg.sequence, msg.payload) == (65535, 0xFFFFFFFF, raw)
    uplink = prepare_lorawan_uplink(packet)
    assert uplink.payload == packet
    assert uplink.fport == 20
    damaged = packet[:-1] + bytes([packet[-1] ^ 1])
    with pytest.raises(ValueError, match="CRC32"):
        prepare_lorawan_uplink(damaged)
    with pytest.raises(ValueError):
        decode_p2p_message(packet[:-4])
    with pytest.raises(ValueError):
        encode_p2p_message(65536, 0, raw)
    with pytest.raises(ValueError, match="FPort"):
        prepare_lorawan_uplink(packet, 0)
    with pytest.raises(ValueError, match="51"):
        prepare_lorawan_uplink(encode_p2p_message(1, 0, bytes(37)))


def test_p2p_measured_tx_time_changes_sensor_energy_as_expected(configs):
    from src.p2p import evaluate_p2p_bridge
    cfg = configs["apriori"]
    short = evaluate_p2p_bridge(cfg, measured_airtime_ms=100)
    long = evaluate_p2p_bridge(cfg, measured_airtime_ms=200)
    radio = cfg["radio_modules"]["EBYTE"]
    mcu = cfg["mcu_configs"]["K1_ESP32"]
    from src.node import calc_mcu_active_current_ma
    expected_delta = (
        radio["supply_voltage_v"] * (110 - 0.005) / radio["power_conversion_efficiency"]
        + short.reference_voltage_v * (calc_mcu_active_current_ma(mcu, 80) - mcu["sleep_modes"]["deep_sleep"]["current_ma"])
    ) * 0.1
    assert long.sensor_cycle_energy_mj - short.sensor_cycle_energy_mj == pytest.approx(expected_delta)
    # При том же числе сообщений время сенсорного TX не меняет цикл LoRaWAN моста.
    assert long.bridge_cycle_energy_mj == pytest.approx(short.bridge_cycle_energy_mj)


@pytest.mark.parametrize("kwargs", [
    {"channel": -1}, {"channel": 84}, {"channel": 1.5}, {"tx_power_dbm": 12},
    {"air_rate_bps": 125000}, {"payload_bytes": 37}, {"tx_period_s": 0},
    {"measured_airtime_ms": -1}, {"measured_airtime_ms": float("nan")},
    {"measured_airtime_ms": 2000, "tx_period_s": 1},
])
def test_p2p_rejects_invalid_or_unimplementable_scenarios(configs, kwargs):
    from src.p2p import evaluate_p2p_bridge
    with pytest.raises(ValueError):
        evaluate_p2p_bridge(configs["apriori"], **kwargs)


def test_p2p_channel_and_bridge_are_independent(configs):
    from src.p2p import evaluate_p2p_bridge
    cfg = configs["apriori"]
    a = evaluate_p2p_bridge(cfg, channel=23)
    b = evaluate_p2p_bridge(cfg, channel=24, bridge_radio_module="DXLR03900", bridge_band_plan="EU868")
    assert b.frequency_hz - a.frequency_hz == 1000000
    assert b.sensor_cycle_energy_mj == pytest.approx(a.sensor_cycle_energy_mj)
    assert b.bridge_lorawan.radio_module == "DXLR03900"
    assert b.bridge_lorawan.band_plan == "EU868"
    assert a.bridge_avg_current_ma > a.sensor_avg_current_ma
    assert a.bridge_battery.lifetime_days < a.sensor_battery.lifetime_days


def test_p2p_sensitivity_affects_link_not_energy(configs):
    from src.p2p import evaluate_p2p_bridge
    good = evaluate_p2p_bridge(configs["apriori"], sensitivity_dbm=-127)
    poor = evaluate_p2p_bridge(configs["apriori"], sensitivity_dbm=-50)
    assert good.p2p_margin_db > 0
    assert poor.p2p_margin_db < 0
    assert not poor.end_to_end_link_feasible
    assert poor.sensor_cycle_energy_mj == good.sensor_cycle_energy_mj


def test_p2p_supports_atmega_sensor(configs):
    from src.p2p import evaluate_p2p_bridge
    r = evaluate_p2p_bridge(configs["apriori"], mcu_config="K2_ATMEGA", cpu_freq_mhz=4)
    assert r.sensor_avg_current_ma > 0
    assert r.bridge_lorawan.mcu_config == "K1_ESP32"


def test_p2p_bridge_controls_do_not_change_sensor_energy(configs):
    from src.p2p import evaluate_p2p_bridge
    cfg = configs["apriori"]
    base = evaluate_p2p_bridge(cfg)
    changed = evaluate_p2p_bridge(
        cfg, bridge_mcu_config="K2_ATMEGA", bridge_cpu_freq_mhz=8,
        bridge_sleep_mode="idle", bridge_wait_mode="active", bridge_device_class="B",
        bridge_ping_periodicity_k=1, bridge_adr_enabled=True,
        bridge_receiver_sensitivity_dbm=-130, bridge_gateway_sensitivity_dbm=-136,
    )
    assert changed.sensor_cycle_energy_mj == pytest.approx(base.sensor_cycle_energy_mj)
    forward = changed.bridge_lorawan
    assert (forward.cpu_freq_mhz, forward.sleep_mode, forward.wait_mode) == (8, "idle", "active")
    assert (forward.device_class, forward.ping_periodicity_k) == ("B", 1)
    assert forward.delta_i_class_b_ma > 0
    assert forward.sf < 9
    assert changed.bridge_cycle_energy_mj == pytest.approx(
        forward.e_cycle_total_mj + changed.bridge_p2p_rx_energy_mj + changed.bridge_processing_energy_mj)


def test_p2p_bridge_class_b_addon_is_not_applied_to_e220_receiver(configs):
    from src.p2p import evaluate_p2p_bridge
    cfg = configs["apriori"]
    a = evaluate_p2p_bridge(cfg)
    b = evaluate_p2p_bridge(cfg, bridge_device_class="B", bridge_ping_periodicity_k=4)
    assert b.bridge_avg_current_ma - a.bridge_avg_current_ma == pytest.approx(b.bridge_lorawan.delta_i_class_b_ma)
    assert b.bridge_p2p_rx_energy_mj == a.bridge_p2p_rx_energy_mj
    assert b.sensor_avg_current_ma == a.sensor_avg_current_ma
    assert b.bridge_battery.lifetime_days < a.bridge_battery.lifetime_days


@pytest.mark.parametrize("kwargs", [
    {"bridge_cpu_freq_mhz": 4}, {"bridge_device_class": "C"}, {"bridge_wait_mode": "invalid"},
])
def test_p2p_rejects_unsupported_bridge_controls(configs, kwargs):
    from src.p2p import evaluate_p2p_bridge
    with pytest.raises(ValueError):
        evaluate_p2p_bridge(configs["apriori"], **kwargs)
