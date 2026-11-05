"""
Модуль моделирования оконечного узла LoRaWAN (п. 2.5, 2.6.1, 2.6.3).
Реализует:
- формулу (2.1): время безубыточности для режимов сна;
- формулу (2.4): линейную зависимость активного тока МК от тактовой частоты и масштабирование времени вычислений;
- формулу (2.3): средний ток за период передачи (с учётом двух цепей питания, опроса датчика и класса B);
- формулу (2.9): полный бюджет энергии за цикл и за сутки с разбивкой по состояниям и подсистемам.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict

from src.states import (
    CLASS_A_ACTIVE_SEQUENCE,
    CycleProfile,
    NodeState,
    PowerCircuit,
    get_supported_band_plans,
    resolve_band_plan,
)
from src.mac import calc_class_a_timings, calc_class_b_current_addon, ClassBMetrics
from src.channel_adr import resolve_adr, AdrConvergenceResult
from src.battery import evaluate_battery_from_config, BatteryLifetimeResult
from src.channel_adr import calc_link_budget
from src.radio import (
    get_radio_compatibility_note,
    module_sensitivity_dbm,
    sensitivity_at_sf,
    validate_radio_band_plan,
)


@dataclass(frozen=True)
class EnergyBudgetResult:
    """Полный результат расчёта энергетического бюджета узла (Таблица 2.2)."""
    # Входные параметры режима
    sf: int
    tx_power_dbm: float
    payload_bytes: int
    tx_period_s: float
    sensor_freq_hz: float
    mcu_config: str
    cpu_freq_mhz: float
    sleep_mode: str
    wait_mode: str
    device_class: str
    ping_periodicity_k: int | None
    radio_module: str
    band_plan: str
    battery_type: str
    radio_supply_voltage_v: float
    receiver_sensitivity_dbm: float  # В опорном режиме выбранного модуля
    rx1_sensitivity_dbm: float
    rx2_sensitivity_dbm: float
    downlink_rx1_margin_db: float
    downlink_rx2_margin_db: float
    downlink_feasible: bool
    beacon_margin_db: float | None
    ping_slot_margin_db: float | None
    class_b_link_feasible: bool | None
    link_feasible: bool             # Uplink и хотя бы одно окно downlink
    model_warnings: tuple[str, ...]

    # Профиль состояний и показатели времени
    profile: CycleProfile
    active_tx_rx_duration_ms: float  # Длительность активной транзакции класса A (t_act), мс
    sleep_duration_s: float          # Время сна за период T_tx, с

    # Заряды и токи (формула 2.3)
    q_tx_rx_radio_mc: float          # Заряд радиоцепи, приведённый к питанию МК с учётом КПД, мКл
    q_tx_rx_mcu_mc: float            # Заряд платы МК за активную транзакцию, мКл
    q_tx_rx_total_mc: float          # Полный заряд радиотранзакции (WU..PP), мКл
    q_sensor_per_tx_period_mc: float # Заряд на опрос датчика за период T_tx, мКл
    q_sleep_per_tx_period_mc: float  # Заряд сна за период T_tx, мКл
    i_sleep_total_ua: float          # Суммарный ток сна (радио + МК + датчик + плата), мкА
    i_avg_class_a_ma: float          # Средний ток в классе A, мА
    delta_i_class_b_ma: float        # Надбавка среднего тока в классе B (формула 2.5), мА
    i_avg_total_ma: float            # Средний ток, эквивалентный на опорной шине питания МК, мА

    # Энергетический бюджет (формула 2.9)
    e_tx_rx_cycle_mj: float          # Энергия одной радиотранзакции (WU..PP), мДж
    e_cycle_total_mj: float          # Полная энергия за период T_tx (с датчиком, сном и классом B), мДж
    e_day_total_j: float             # Суточное потребление энергии E_day, Дж

    # Доли составляющих в полном бюджете энергии (%)
    share_radio_pct: float           # Доля цепи радиомодуля, %
    share_mcu_board_pct: float       # Доля цепи МК и платы (включая датчик), %
    share_sleep_pct: float           # Доля режима сна (обе цепи), %
    share_wait_windows_pct: float    # Доля интервалов ожидания WAIT_RX1 + WAIT_RX2 (для Г5), %
    share_rx2_in_tx_rx_pct: float    # Доля окна RX2 в энергии радиотранзакции (для Г1), %
    share_sensor_pct: float          # Доля опроса датчика BME280, %
    share_class_b_pct: float         # Доля надбавки класса B (маяки + ping-слоты, для Г3), %

    # Срок службы батареи и ADR
    battery: BatteryLifetimeResult
    adr_info: AdrConvergenceResult
    class_b_info: ClassBMetrics | None


def calc_sleep_breakeven_time_s(
    e_wu_shallow_uj: float,
    e_wu_deep_uj: float,
    i_sleep_shallow_ua: float,
    i_sleep_deep_ua: float,
    voltage_v: float = 3.3,
) -> float:
    """
    Формула (2.1): время безубыточности t_be (в секундах) — минимальная длительность сна,
    при которой переход в более глубокий режим сна выгоден с учётом энергии выхода из него:
    t_be = (E_wu_deep - E_wu_shallow) / (U * (I_sleep_shallow - I_sleep_deep)).

    Так как числитель в мкДж, а ток в мкА, отношение мкДж / (В * мкА) = мкДж / мкВт = секунды.
    """
    delta_i_ua = i_sleep_shallow_ua - i_sleep_deep_ua
    if delta_i_ua <= 0:
        raise ValueError("Ток менее глубокого сна должен быть строго больше тока глубокого сна.")
    delta_e_uj = max(0.0, e_wu_deep_uj - e_wu_shallow_uj)
    return delta_e_uj / (voltage_v * delta_i_ua)


def calc_mcu_active_current_ma(
    mcu_cfg: Dict[str, Any],
    cpu_freq_mhz: float,
    use_exact_table_if_present: bool = True,
) -> float:
    """
    Формула (2.4): ток микроконтроллера в активном состоянии в зависимости от тактовой частоты
    I_mcu(f) = I_0 + k_f * f, плюс фоновый ток платы и гибернации второго МК (если есть).
    """
    freq_table = mcu_cfg.get("active_current_by_freq_ma", {})
    freq_int = int(round(cpu_freq_mhz))
    if use_exact_table_if_present and freq_int in freq_table:
        core_ma = float(freq_table[freq_int])
    else:
        i0 = float(mcu_cfg["i0_ma"])
        kf = float(mcu_cfg["kf_ma_per_mhz"])
        core_ma = i0 + kf * float(cpu_freq_mhz)

    board_q_ma = float(mcu_cfg.get("board_quiescent_current_ma", 0.0))
    co_mcu_ma = float(mcu_cfg.get("co_mcu_hibernation_ma", 0.0))
    return core_ma + board_q_ma + co_mcu_ma


def get_sleep_wakeup_cost(mode_cfg: Dict[str, Any], voltage_v: float) -> tuple[float, float]:
    """
    Возвращает длительность (мс) и заряд (мКл) выхода МК из режима сна по параметрам sleep_modes:
    t_wu = wakeup_time_ms, Q_wu = E_wu / U (мкДж / В = мкКл, делением на 1000 переводится в мКл).
    """
    t_wu_ms = float(mode_cfg.get("wakeup_time_ms", 0.0))
    q_wu_mc = float(mode_cfg.get("wakeup_energy_uj", 0.0)) / voltage_v / 1000.0
    return t_wu_ms, q_wu_mc


def get_radio_tx_current_ma(radio_cfg: Dict[str, Any], tx_power_dbm: float) -> float:
    """Возвращает ток радиомодуля при передаче на заданной мощности P_tx (с линейной интерполяцией)."""
    if not math.isfinite(tx_power_dbm) or not (
        float(radio_cfg.get("min_tx_power_dbm", -math.inf)) <= tx_power_dbm
        <= float(radio_cfg.get("max_tx_power_dbm", math.inf))
    ):
        raise ValueError("Мощность передачи вне допустимого диапазона радиомодуля.")
    p_table = {float(k): float(v) for k, v in radio_cfg["tx_current_by_p_dbm"].items()}
    if radio_cfg.get("tx_current_model") == "constant_estimate":
        return next(iter(p_table.values()))
    if tx_power_dbm in p_table:
        return p_table[tx_power_dbm]
    sorted_p = sorted(p_table.keys())
    if tx_power_dbm <= sorted_p[0]:
        return p_table[sorted_p[0]]
    if tx_power_dbm >= sorted_p[-1]:
        return p_table[sorted_p[-1]]
    for i in range(len(sorted_p) - 1):
        p1, p2 = sorted_p[i], sorted_p[i + 1]
        if p1 <= tx_power_dbm <= p2:
            ratio = (tx_power_dbm - p1) / (p2 - p1)
            return p_table[p1] + ratio * (p_table[p2] - p_table[p1])
    return p_table[sorted_p[0]]


def evaluate_node_energy_budget(
    config: Dict[str, Any],
    sf: int | None = None,
    adr_enabled: bool | None = None,
    distance_km: float | None = None,
    payload_bytes: int | None = None,
    sensor_freq_hz: float | None = None,
    mcu_config: str | None = None,
    cpu_freq_mhz: float | None = None,
    sleep_mode: str | None = None,
    wait_mode: str | None = None,
    device_class: str | None = None,
    ping_periodicity_k: int | None = None,
    battery_type: str | None = None,
    tx_period_s: float | None = None,
    radio_module: str | None = None,
    band_plan: str | None = None,
    receiver_sensitivity_dbm: float | None = None,
    gateway_sensitivity_dbm: float | None = None,
) -> EnergyBudgetResult:
    """
    Рассчитывает полный профиль состояний, средний ток, энергетический бюджет и срок службы узла.
    Все незаданные параметры автоматически берутся из секции `experiment_grid.baseline` конфига.
    receiver_sensitivity_dbm задаёт порог узла в опорных SF/BW его даташита (downlink).
    gateway_sensitivity_dbm задаёт порог шлюза в опорных SF/BW плана (uplink).
    Суммарные токи/заряды приведены к system.supply_voltage_v; профиль хранит токи физических шин.
    """
    base = config["experiment_grid"]["baseline"]

    sf_val = int(base["sf"] if sf is None else sf)
    adr_val = bool(base["adr_enabled"] if adr_enabled is None else adr_enabled)
    dist_val = float(base["distance_km"] if distance_km is None else distance_km)
    pl_val = int(base["payload_bytes"] if payload_bytes is None else payload_bytes)
    f_sens_val = float(base["sensor_freq_hz"] if sensor_freq_hz is None else sensor_freq_hz)
    mcu_key = str(base["mcu_config"] if mcu_config is None else mcu_config)
    mcu_cfg = config["mcu_configs"][mcu_key]

    if cpu_freq_mhz is None:
        f_cpu_val = float(base["cpu_freq_mhz"] if mcu_key == base["mcu_config"] else mcu_cfg["base_freq_mhz"])
    else:
        f_cpu_val = float(cpu_freq_mhz)

    if sleep_mode is None:
        if mcu_key == "K1_ESP32":
            sleep_key = str(base.get("sleep_mode", "deep_sleep"))
        else:
            sleep_key = "power_down"
    else:
        sleep_key = str(sleep_mode)

    wait_key = str(base.get("wait_mode", "light_sleep") if wait_mode is None else wait_mode)
    cls_val = str(base["device_class"] if device_class is None else device_class).upper()
    k_val = base["ping_periodicity_k"] if ping_periodicity_k is None else ping_periodicity_k
    bat_key = str(base["battery_type"] if battery_type is None else battery_type)
    t_tx_s = float(base["tx_period_s"] if tx_period_s is None else tx_period_s)
    rad_key = str(base.get("radio_module", "DXLR03900") if radio_module is None else radio_module)
    band_plan_key, band_plan_cfg = resolve_band_plan(config, band_plan)
    sys_cfg = band_plan_cfg["system"]
    ch_cfg = band_plan_cfg["channel_adr"]
    class_b_cfg = band_plan_cfg["class_b"]

    voltage_v = float(sys_cfg["supply_voltage_v"])
    radio_modules = config["radio_modules"]
    if rad_key not in radio_modules:
        available_modules = ", ".join(sorted(map(str, radio_modules.keys())))
        raise ValueError(
            f"Неизвестный радиомодуль '{rad_key}'. Доступные варианты: {available_modules}."
        )
    radio_cfg = radio_modules[rad_key]
    if radio_cfg.get("lorawan_support") == "unsupported_stock_firmware":
        raise ValueError(get_radio_compatibility_note(radio_cfg))
    supported_band_plans = get_supported_band_plans(config, rad_key)
    if band_plan_key not in supported_band_plans:
        raise ValueError(
            f"Частотный план '{band_plan_key}' не поддерживается радиомодулем '{rad_key}'. "
            f"Совместимые варианты: {', '.join(supported_band_plans)}."
        )
    validate_radio_band_plan(radio_cfg, band_plan_cfg)
    radio_voltage_v = float(radio_cfg.get("supply_voltage_v", voltage_v))
    v_min, v_max = radio_cfg.get("supply_voltage_range_v", [radio_voltage_v, radio_voltage_v])
    efficiency = float(radio_cfg.get("power_conversion_efficiency", 1.0))
    if not v_min <= radio_voltage_v <= v_max or not 0 < efficiency <= 1:
        raise ValueError("Некорректное напряжение питания радиомодуля или КПД преобразователя.")
    radio_charge_scale = radio_voltage_v / voltage_v / efficiency
    receiver_ref_dbm = float(radio_cfg["sensitivity_dbm"] if receiver_sensitivity_dbm is None else receiver_sensitivity_dbm)
    gateway_ref_dbm = ch_cfg.get("gateway_sensitivity_dbm") if gateway_sensitivity_dbm is None else gateway_sensitivity_dbm
    gateway_thresholds = None if gateway_ref_dbm is None else {
        s: sensitivity_at_sf(float(gateway_ref_dbm), s,
                             int(ch_cfg.get("gateway_sensitivity_reference_sf", 12)),
                             float(sys_cfg["bandwidth_khz"]),
                             float(ch_cfg.get("gateway_sensitivity_reference_bandwidth_khz", 125)))
        for s in range(7, 13)
    }
    # L0 относится к конкретной частоте. Перенос между диапазонами — оценка 20 lg(f/f_ref).
    frequency_hz = float(sys_cfg["frequency_hz"])
    reference_frequency_hz = float(ch_cfg.get("path_loss_reference_frequency_hz", frequency_hz))
    ref_loss_db = float(ch_cfg["ref_loss_db"]) + 20.0 * math.log10(frequency_hz / reference_frequency_hz)
    max_power = min(float(radio_cfg["max_tx_power_dbm"]),
                    float(sys_cfg.get("max_eirp_dbm", radio_cfg["max_tx_power_dbm"]))
                    - float(ch_cfg.get("tx_antenna_gain_dbi", 0.0)))
    start_power = min(float(ch_cfg["tx_power_dbm"]), max_power)
    min_power = max(float(radio_cfg.get("min_tx_power_dbm", 2)), float(ch_cfg.get("min_tx_power_dbm", 2)))
    if min_power > start_power:
        raise ValueError("Мощность плана ниже минимальной мощности выбранного радиомодуля.")
    sens_cfg = config["sensor"]

    # 1. Расчёт канала и ADR (формулы 2.6, 2.7)
    snr_thresh = {int(k): float(v) for k, v in ch_cfg["snr_thresholds_db"].items()}
    adr_res = resolve_adr(
        distance_km=dist_val,
        adr_enabled=adr_val,
        fixed_sf=sf_val,
        initial_sf_when_adr=12,
        tx_power_dbm=start_power,
        min_tx_power_dbm=min_power,
        tx_power_step_db=float(ch_cfg.get("tx_power_step_db", 2.0)),
        antenna_gain_db=float(ch_cfg["antenna_gain_db"]),
        ref_distance_km=float(ch_cfg["ref_distance_km"]),
        ref_loss_db=ref_loss_db,
        path_loss_exponent=float(ch_cfg["path_loss_exponent"]),
        bw_khz=float(sys_cfg["bandwidth_khz"]),
        gateway_noise_figure_db=float(ch_cfg["gateway_noise_figure_db"]),
        adr_margin_db=float(ch_cfg["adr_margin_db"]),
        snr_thresholds_db=snr_thresh,
        receiver_sensitivity_by_sf_dbm=gateway_thresholds,
    )
    actual_sf = adr_res.final_sf
    actual_p_tx = adr_res.final_tx_power_dbm

    # 2. Временные диаграммы физического и MAC-уровней (формула 2.2)
    timings = calc_class_a_timings(
        sf_tx=actual_sf,
        payload_bytes=pl_val,
        bw_khz=float(sys_cfg["bandwidth_khz"]),
        coding_rate=int(sys_cfg["coding_rate"]),
        preamble_symbols=int(sys_cfg["preamble_symbols"]),
        lorawan_header_bytes=int(sys_cfg["lorawan_header_bytes"]),
        n_sym_rx=int(sys_cfg["rx_window_symbols"]),
        rx1_delay_ms=float(sys_cfg["rx1_delay_ms"]),
        rx2_delay_ms=float(sys_cfg["rx2_delay_ms"]),
        rx1_dr_offset=int(sys_cfg["rx1_dr_offset"]),
        rx2_sf=int(sys_cfg["rx2_sf"]),
    )
    rx1_sensitivity = module_sensitivity_dbm(radio_cfg, timings.sf_rx1, float(sys_cfg["bandwidth_khz"]), receiver_ref_dbm)
    rx2_sensitivity = module_sensitivity_dbm(radio_cfg, timings.sf_rx2, float(sys_cfg["bandwidth_khz"]), receiver_ref_dbm)
    gateway_power = float(ch_cfg.get("gateway_tx_power_dbm", start_power))
    # При взаимном канале потери одинаковы; для RX2 учитывается другая частота.
    downlink = calc_link_budget(
        distance_km=dist_val, tx_power_dbm=gateway_power,
        antenna_gain_db=float(ch_cfg["antenna_gain_db"]),
        ref_distance_km=float(ch_cfg["ref_distance_km"]), ref_loss_db=ref_loss_db,
        path_loss_exponent=float(ch_cfg["path_loss_exponent"]),
    )
    rx1_margin = downlink.rssi_dbm - rx1_sensitivity
    rx2_frequency_hz = float(sys_cfg.get("rx2_frequency_hz", frequency_hz))
    rx2_margin = downlink.rssi_dbm - 20.0 * math.log10(rx2_frequency_hz / frequency_hz) - rx2_sensitivity
    downlink_feasible = rx1_margin >= 0 or rx2_margin >= 0
    beacon_margin = ping_margin = None
    class_b_link_feasible = None
    if cls_val == "B":
        def class_b_rx_margin(sf_key: str, freq_key: str) -> float:
            threshold = module_sensitivity_dbm(
                radio_cfg, int(class_b_cfg[sf_key]), float(sys_cfg["bandwidth_khz"]), receiver_ref_dbm
            )
            frequency = float(class_b_cfg.get(freq_key, frequency_hz))
            return downlink.rssi_dbm - 20.0 * math.log10(frequency / frequency_hz) - threshold

        beacon_margin = class_b_rx_margin("beacon_sf", "beacon_frequency_hz")
        ping_margin = class_b_rx_margin("ping_slot_sf", "ping_slot_frequency_hz")
        class_b_link_feasible = beacon_margin >= 0 and ping_margin >= 0
    model_warnings = []
    if radio_cfg.get("lorawan_support") == "requires_custom_firmware":
        model_warnings.append(get_radio_compatibility_note(radio_cfg))
    if radio_cfg.get("tx_current_note"):
        model_warnings.append(str(radio_cfg["tx_current_note"]))
    if start_power < float(ch_cfg["tx_power_dbm"]):
        model_warnings.append(f"Мощность ограничена до {start_power:g} дБм пределом модуля/плана.")
    duty_limit = sys_cfg.get("duty_cycle_limit")
    if duty_limit and timings.tx_ms / 1000.0 / t_tx_s > float(duty_limit):
        model_warnings.append("Выбранные SF, нагрузка и период превышают duty cycle плана; энергия рассчитана для заданного трафика без планировщика пауз.")

    # 3. Токи цепей и масштабирование времени вычислений МК от частоты (формула 2.4)
    base_freq = float(mcu_cfg["base_freq_mhz"])
    freq_scale = base_freq / f_cpu_val
    wu_ms = float(mcu_cfg["wu_duration_base_ms"]) * freq_scale
    pp_ms = float(mcu_cfg["pp_duration_base_ms"]) * freq_scale

    i_mcu_active_ma = calc_mcu_active_current_ma(mcu_cfg, f_cpu_val)
    board_q_ma = float(mcu_cfg.get("board_quiescent_current_ma", 0.0))
    co_mcu_ma = float(mcu_cfg.get("co_mcu_hibernation_ma", 0.0))
    i_sens_sleep_ma = float(sens_cfg["sleep_current_ma"])

    # Ток платы МК в паузах ожидания окон приёма WAIT_RX1 и WAIT_RX2 (гипотеза Г5)
    if wait_key == "active":
        i_mcu_wait_ma = i_mcu_active_ma + i_sens_sleep_ma
    else:
        # Неглубокий сон: light_sleep для ESP32 или idle для ATmega
        shallow_key = "light_sleep" if "light_sleep" in mcu_cfg["sleep_modes"] else "idle"
        i_mcu_wait_ma = (
            float(mcu_cfg["sleep_modes"][shallow_key]["current_ma"])
            + board_q_ma
            + co_mcu_ma
            + i_sens_sleep_ma
        )

    # Ток основного режима сна
    i_mcu_deep_sleep_ma = (
        float(mcu_cfg["sleep_modes"][sleep_key]["current_ma"])
        + board_q_ma
        + co_mcu_ma
        + i_sens_sleep_ma
    )

    # Токи радиомодуля
    i_rad_sleep_ma = float(radio_cfg["sleep_current_ma"])
    i_rad_standby_ma = float(radio_cfg["standby_current_ma"])
    i_rad_rx_ma = float(radio_cfg["rx_current_ma"])
    i_rad_tx_ma = get_radio_tx_current_ma(radio_cfg, actual_p_tx)

    # Выход из основного режима сна (формула 2.1): каждое пробуждение стоит E_wu и t_wu,
    # не зависящих от тактовой частоты (для ESP32 deep sleep — перезагрузка и инициализация)
    t_wake_ms, q_wake_mc = get_sleep_wakeup_cost(mcu_cfg["sleep_modes"][sleep_key], voltage_v)

    # 4. Заполнение профиля состояний по двум цепям питания (рисунок 2.3)
    profile = CycleProfile(voltage_v=voltage_v, radio_voltage_v=radio_voltage_v)

    def q_ref(state: NodeState, circuit: PowerCircuit | None = None) -> float:
        """Заряд, приведённый к питанию платы, с учётом КПД радиоцепи."""
        if circuit is None:
            return sum(q_ref(state, c) for c in PowerCircuit)
        q = profile.get_state_charge_mc(state, circuit)
        return q * radio_charge_scale if circuit == PowerCircuit.RADIO else q
    # WU = выход из сна + подготовка кадра (wu_ms масштабируется по частоте, формула 2.4)
    wu_total_ms = t_wake_ms + wu_ms
    q_wu_radio_mc = (i_rad_sleep_ma * t_wake_ms + i_rad_standby_ma * wu_ms) / 1000.0
    q_wu_mcu_mc = q_wake_mc + (i_mcu_active_ma + i_sens_sleep_ma) * wu_ms / 1000.0
    profile.add_state(
        NodeState.WU,
        wu_total_ms,
        1000.0 * q_wu_radio_mc / wu_total_ms,
        1000.0 * q_wu_mcu_mc / wu_total_ms,
    )
    profile.add_state(NodeState.TX, timings.tx_ms, i_rad_tx_ma, i_mcu_active_ma + i_sens_sleep_ma)
    profile.add_state(NodeState.WAIT_RX1, timings.wait_rx1_ms, i_rad_sleep_ma, i_mcu_wait_ma)
    profile.add_state(NodeState.RX1, timings.rx1_ms, i_rad_rx_ma, i_mcu_active_ma + i_sens_sleep_ma)
    profile.add_state(NodeState.WAIT_RX2, timings.wait_rx2_ms, i_rad_sleep_ma, i_mcu_wait_ma)
    profile.add_state(NodeState.RX2, timings.rx2_ms, i_rad_rx_ma, i_mcu_active_ma + i_sens_sleep_ma)
    profile.add_state(NodeState.PP, pp_ms, i_rad_standby_ma, i_mcu_active_ma + i_sens_sleep_ma)

    # Состояние опроса датчика BME280 (forced mode)
    sens_dur_ms = float(sens_cfg["measurement_duration_ms"])
    i_sens_act_ma = float(sens_cfg["active_current_ma"])
    profile.add_state(
        NodeState.SENSOR,
        sens_dur_ms,
        i_rad_sleep_ma,  # Радио продолжает потреблять ток сна во время опроса
        i_mcu_wait_ma + i_sens_act_ma,
    )

    # 5. Суммирование зарядов за период передачи T_tx (формула 2.3)
    t_act_ms = profile.get_active_duration_ms()
    t_act_s = t_act_ms / 1000.0

    # Число измерений датчика за период передачи T_tx. Одно измерение выполняется в пробуждении
    # для передачи, остальные требуют отдельного выхода МК из сна (E_wu, t_wu)
    n_sensor_reads = f_sens_val * t_tx_s
    n_sensor_wakeups = max(0.0, n_sensor_reads - 1.0)
    t_sensor_total_s = (n_sensor_reads * sens_dur_ms + n_sensor_wakeups * t_wake_ms) / 1000.0
    q_sensor_One_mc = q_ref(NodeState.SENSOR, PowerCircuit.MCU_BOARD)
    q_sensor_mcu_mc = n_sensor_reads * q_sensor_One_mc + n_sensor_wakeups * q_wake_mc
    q_sensor_radio_mc = i_rad_sleep_ma * t_sensor_total_s * radio_charge_scale
    q_sensor_period_mc = q_sensor_mcu_mc + q_sensor_radio_mc

    # Длительность сна за период передачи
    t_sleep_s = max(0.0, t_tx_s - t_act_s - t_sensor_total_s)
    profile.add_state(NodeState.SLEEP, t_sleep_s * 1000.0, i_rad_sleep_ma, i_mcu_deep_sleep_ma)

    q_tx_rx_radio_mc = sum(
        q_ref(st, PowerCircuit.RADIO) for st in CLASS_A_ACTIVE_SEQUENCE
    )
    q_tx_rx_mcu_mc = sum(
        q_ref(st, PowerCircuit.MCU_BOARD) for st in CLASS_A_ACTIVE_SEQUENCE
    )
    q_tx_rx_total_mc = q_tx_rx_radio_mc + q_tx_rx_mcu_mc

    q_sleep_radio_mc = q_ref(NodeState.SLEEP, PowerCircuit.RADIO)
    q_sleep_mcu_mc = q_ref(NodeState.SLEEP, PowerCircuit.MCU_BOARD)
    q_sleep_total_mc = q_sleep_radio_mc + q_sleep_mcu_mc

    i_sleep_total_ma = i_rad_sleep_ma * radio_charge_scale + i_mcu_deep_sleep_ma
    i_avg_class_a_ma = (q_tx_rx_total_mc + q_sensor_period_mc + q_sleep_total_mc) / t_tx_s

    # 6. Учёт класса B (формула 2.5)
    class_b_info: ClassBMetrics | None = None
    delta_i_b_ma = 0.0
    q_class_b_radio_per_tx_mc = 0.0
    q_class_b_mcu_per_tx_mc = 0.0

    if cls_val == "B":
        k_int = 4 if k_val is None else int(k_val)
        b_dur_ms = float(class_b_cfg["beacon_rx_duration_ms"])
        p_dur_ms = float(class_b_cfg["ping_slot_duration_ms"])

        # Каждый приём маяка и каждый ping-слот начинается с выхода МК из основного режима сна
        for st_b, rx_ms in ((NodeState.BEACON_RX, b_dur_ms), (NodeState.PING_SLOT, p_dur_ms)):
            dur_ms = t_wake_ms + rx_ms
            q_rad_mc = (i_rad_sleep_ma * t_wake_ms + i_rad_rx_ma * rx_ms) / 1000.0
            q_mcu_mc = q_wake_mc + (i_mcu_active_ma + i_sens_sleep_ma) * rx_ms / 1000.0
            profile.add_state(st_b, dur_ms, 1000.0 * q_rad_mc / dur_ms, 1000.0 * q_mcu_mc / dur_ms)

        # Заряд события сверх тока сна: время маяков и ping-слотов уже входит в состояние SLEEP
        sleep_current_by_circuit = {PowerCircuit.RADIO: i_rad_sleep_ma * radio_charge_scale, PowerCircuit.MCU_BOARD: i_mcu_deep_sleep_ma}

        def net_charge_mc(st_b: NodeState, circuit: PowerCircuit) -> float:
            dur_s = profile.states[st_b][circuit].duration_s
            return q_ref(st_b, circuit) - sleep_current_by_circuit[circuit] * dur_s

        q_b_mc = sum(net_charge_mc(NodeState.BEACON_RX, c) for c in PowerCircuit)
        q_p_mc = sum(net_charge_mc(NodeState.PING_SLOT, c) for c in PowerCircuit)

        class_b_info = calc_class_b_current_addon(
            periodicity_k=k_int,
            q_beacon_mc=q_b_mc,
            q_ping_mc=q_p_mc,
            beacon_period_s=float(class_b_cfg["beacon_period_s"]),
            beacon_window_s=float(class_b_cfg.get("beacon_window_s", 122.88)),
        )
        delta_i_b_ma = class_b_info.delta_i_class_b_ma

        # Разделение заряда класса B по цепям радио и МК за период T_tx
        scale_b = t_tx_s / float(class_b_cfg["beacon_period_s"])
        slots = class_b_info.slots_per_beacon
        q_class_b_radio_per_tx_mc = scale_b * (
            net_charge_mc(NodeState.BEACON_RX, PowerCircuit.RADIO)
            + slots * net_charge_mc(NodeState.PING_SLOT, PowerCircuit.RADIO)
        )
        q_class_b_mcu_per_tx_mc = scale_b * (
            net_charge_mc(NodeState.BEACON_RX, PowerCircuit.MCU_BOARD)
            + slots * net_charge_mc(NodeState.PING_SLOT, PowerCircuit.MCU_BOARD)
        )

    i_avg_total_ma = i_avg_class_a_ma + delta_i_b_ma

    # 7. Энергетический бюджет и доли составляющих (формула 2.9)
    q_cycle_total_mc = i_avg_total_ma * t_tx_s
    e_tx_rx_cycle_mj = voltage_v * q_tx_rx_total_mc
    e_cycle_total_mj = voltage_v * q_cycle_total_mc
    # Суточная энергия в Дж: U (В) * I_avg (мА) * 86400 (с) / 1000 = Дж
    e_day_total_j = voltage_v * i_avg_total_ma * 86.4

    q_radio_all_mc = q_tx_rx_radio_mc + q_sensor_radio_mc + q_sleep_radio_mc + q_class_b_radio_per_tx_mc
    q_mcu_all_mc = q_tx_rx_mcu_mc + q_sensor_mcu_mc + q_sleep_mcu_mc + q_class_b_mcu_per_tx_mc
    q_wait_mc = q_ref(NodeState.WAIT_RX1) + q_ref(NodeState.WAIT_RX2)
    q_rx2_mc = q_ref(NodeState.RX2)
    q_class_b_total_mc = delta_i_b_ma * t_tx_s

    share_radio_pct = 100.0 * q_radio_all_mc / q_cycle_total_mc
    share_mcu_pct = 100.0 * q_mcu_all_mc / q_cycle_total_mc
    share_sleep_pct = 100.0 * q_sleep_total_mc / q_cycle_total_mc
    share_wait_pct = 100.0 * q_wait_mc / q_cycle_total_mc
    share_rx2_in_tx_rx_pct = 100.0 * q_rx2_mc / q_tx_rx_total_mc
    share_sensor_pct = 100.0 * q_sensor_period_mc / q_cycle_total_mc
    share_class_b_pct = 100.0 * q_class_b_total_mc / q_cycle_total_mc

    # 8. Оценка срока службы элемента питания (формула 2.8)
    bat_res = evaluate_battery_from_config(config, bat_key, i_avg_total_ma)

    return EnergyBudgetResult(
        sf=actual_sf,
        tx_power_dbm=actual_p_tx,
        payload_bytes=pl_val,
        tx_period_s=t_tx_s,
        sensor_freq_hz=f_sens_val,
        mcu_config=mcu_key,
        cpu_freq_mhz=f_cpu_val,
        sleep_mode=sleep_key,
        wait_mode=wait_key,
        device_class=cls_val,
        ping_periodicity_k=k_val if cls_val == "B" else None,
        radio_module=rad_key,
        band_plan=band_plan_key,
        battery_type=bat_key,
        radio_supply_voltage_v=radio_voltage_v,
        receiver_sensitivity_dbm=receiver_ref_dbm,
        rx1_sensitivity_dbm=rx1_sensitivity,
        rx2_sensitivity_dbm=rx2_sensitivity,
        downlink_rx1_margin_db=rx1_margin,
        downlink_rx2_margin_db=rx2_margin,
        downlink_feasible=downlink_feasible,
        beacon_margin_db=beacon_margin,
        ping_slot_margin_db=ping_margin,
        class_b_link_feasible=class_b_link_feasible,
        link_feasible=adr_res.link_feasible and downlink_feasible and class_b_link_feasible is not False,
        model_warnings=tuple(model_warnings),
        profile=profile,
        active_tx_rx_duration_ms=t_act_ms,
        sleep_duration_s=t_sleep_s,
        q_tx_rx_radio_mc=q_tx_rx_radio_mc,
        q_tx_rx_mcu_mc=q_tx_rx_mcu_mc,
        q_tx_rx_total_mc=q_tx_rx_total_mc,
        q_sensor_per_tx_period_mc=q_sensor_period_mc,
        q_sleep_per_tx_period_mc=q_sleep_total_mc,
        i_sleep_total_ua=i_sleep_total_ma * 1000.0,
        i_avg_class_a_ma=i_avg_class_a_ma,
        delta_i_class_b_ma=delta_i_b_ma,
        i_avg_total_ma=i_avg_total_ma,
        e_tx_rx_cycle_mj=e_tx_rx_cycle_mj,
        e_cycle_total_mj=e_cycle_total_mj,
        e_day_total_j=e_day_total_j,
        share_radio_pct=share_radio_pct,
        share_mcu_board_pct=share_mcu_pct,
        share_sleep_pct=share_sleep_pct,
        share_wait_windows_pct=share_wait_pct,
        share_rx2_in_tx_rx_pct=share_rx2_in_tx_rx_pct,
        share_sensor_pct=share_sensor_pct,
        share_class_b_pct=share_class_b_pct,
        battery=bat_res,
        adr_info=adr_res,
        class_b_info=class_b_info,
    )
