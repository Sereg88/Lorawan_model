"""
Модуль вычислительного эксперимента, анализа чувствительности и ПФЭ 2^3 (п. 2.6.3, 2.8.1, 2.8.3).
Реализует:
- полный факторный перебор уровней Таблицы 2.1 с экспортом в DataFrame / CSV;
- формулу (2.11): коэффициенты эластичности (чувствительности) и относительного влияния факторов;
- Таблицу 2.6 и формулу (2.12): расчёт коэффициентов регрессии ПФЭ 2^3 для отклика y = lg(I_avg)
  с проверкой по критериям Кохрена и Стьюдента.
"""

from dataclasses import dataclass
import itertools
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence
import numpy as np
import pandas as pd
from scipy import stats

from src.node import evaluate_node_energy_budget
from src.p2p import ENVELOPE_BYTES, P2PBridgeResult, evaluate_p2p_bridge
from src.states import NodeState, PowerCircuit


def p2p_result_tables(result: P2PBridgeResult) -> dict[str, pd.DataFrame]:
    """Таблицы одного маршрута; энергии приведены к опорной шине с учётом КПД."""
    forward = result.bridge_lorawan
    sensor = pd.DataFrame([
        {"state": p.name, "duration_ms": p.duration_ms, "radio_mj": p.radio_energy_mj,
         "mcu_sensor_mj": p.mcu_energy_mj, "total_mj": p.energy_mj,
         "equivalent_current_ma": 1000 * p.energy_mj / result.reference_voltage_v / p.duration_ms
         if p.duration_ms else 0}
        for p in result.sensor_phases
    ])
    bridge = pd.DataFrame([
        {"component": "LoRaWAN + MCU", "energy_mj": forward.e_cycle_total_mj},
        {"component": "E220 continuous RX", "energy_mj": result.bridge_p2p_rx_energy_mj},
        {"component": "Bridge processing above sleep", "energy_mj": result.bridge_processing_energy_mj},
    ])
    # Профиль LoRaWAN показывает физические токи шин. Полный бюджет моста — выше:
    # SENSOR отключён, события класса B и обработка не складываются как линейные фазы.
    bridge_states = pd.DataFrame([
        {"state": state.value, "duration_ms": circuits[PowerCircuit.RADIO].duration_ms,
         "radio_current_ma": circuits[PowerCircuit.RADIO].current_ma,
         "mcu_current_ma": circuits[PowerCircuit.MCU_BOARD].current_ma}
        for state, circuits in forward.profile.states.items()
    ])
    links = pd.DataFrame([
        {"link": "P2P uplink", "margin_db": result.p2p_margin_db},
        {"link": "LoRaWAN uplink", "margin_db": forward.adr_info.link_margin_db},
        {"link": "LoRaWAN RX1", "margin_db": forward.downlink_rx1_margin_db},
        {"link": "LoRaWAN RX2", "margin_db": forward.downlink_rx2_margin_db},
    ] + ([
        {"link": "LoRaWAN beacon", "margin_db": forward.beacon_margin_db},
        {"link": "LoRaWAN ping", "margin_db": forward.ping_slot_margin_db},
    ] if forward.device_class == "B" else []))
    summary = pd.DataFrame([{
        "route_type": "lora_p2p_bridge", "channel": result.channel,
        "frequency_hz": result.frequency_hz, "parameter_profile": result.parameter_profile,
        "air_rate_bps": result.air_rate_bps, "tx_power_dbm": result.tx_power_dbm,
        "airtime_ms": result.airtime_ms, "airtime_source": result.airtime_source,
        "payload_bytes": result.payload_bytes, "forwarded_bytes": result.forwarded_bytes,
        "forward_fport": result.forward_fport, "tx_period_s": result.tx_period_s,
        "reference_voltage_v": result.reference_voltage_v,
        "sensor_avg_ma": result.sensor_avg_current_ma,
        "sensor_cycle_mj": result.sensor_cycle_energy_mj,
        "sensor_lifetime_days": result.sensor_battery.lifetime_days,
        "bridge_avg_ma": result.bridge_avg_current_ma,
        "bridge_cycle_mj": result.bridge_cycle_energy_mj,
        "bridge_lifetime_days": result.bridge_battery.lifetime_days,
        "battery": result.sensor_battery.battery_key,
        "bridge_radio": forward.radio_module, "bridge_plan": forward.band_plan,
        "bridge_class": forward.device_class, "bridge_sf": forward.sf,
        "bridge_tx_power_dbm": forward.tx_power_dbm,
        "p2p_margin_db": result.p2p_margin_db,
        "end_to_end_link_feasible": result.end_to_end_link_feasible,
        "assumptions": " | ".join(result.warnings),
    }])
    return {"summary": summary, "sensor_phases": sensor, "bridge_energy": bridge,
            "bridge_states": bridge_states, "link_margins": links}


def compute_p2p_analysis(config: dict, parameters: dict, section: str) -> dict[str, pd.DataFrame]:
    """Срезы модели P2P вокруг текущей точки, с отдельными откликами двух устройств.

    Невместившиеся в период точки сохраняются с error и пустыми откликами.
    Измерение TX относится только к выбранным payload/rate: их перебор отключён.
    """
    base = evaluate_p2p_bridge(config, **parameters)
    sensor_mcu = parameters.get("mcu_config", "K1_ESP32")
    bridge_mcu = parameters.get("bridge_mcu_config", "K1_ESP32")
    deep = "deep_sleep" if sensor_mcu == "K1_ESP32" else "power_down"
    shallow = "light_sleep" if sensor_mcu == "K1_ESP32" else "idle"
    sensor_cpu = sorted(config["mcu_configs"][sensor_mcu]["active_current_by_freq_ma"])
    bridge_cpu = sorted(config["mcu_configs"][bridge_mcu]["active_current_by_freq_ma"])

    def point(labels, **overrides):
        try:
            r = evaluate_p2p_bridge(config, **{**parameters, **overrides})
        except ValueError as err:
            return {**labels, "error": str(err), "sensor_avg_ma": np.nan, "bridge_avg_ma": np.nan,
                    "sensor_lifetime_days": np.nan, "bridge_lifetime_days": np.nan}
        return {**labels, "error": "", "sensor_avg_ma": r.sensor_avg_current_ma,
                "bridge_avg_ma": r.bridge_avg_current_ma,
                "sensor_lifetime_days": r.sensor_battery.lifetime_days,
                "bridge_lifetime_days": r.bridge_battery.lifetime_days,
                "sensor_cycle_mj": r.sensor_cycle_energy_mj, "bridge_cycle_mj": r.bridge_cycle_energy_mj,
                "airtime_ms": r.airtime_ms, "airtime_source": r.airtime_source,
                "bridge_sf": r.bridge_lorawan.sf, "bridge_tx_power_dbm": r.bridge_lorawan.tx_power_dbm,
                "bridge_transaction_mj": r.bridge_lorawan.e_tx_rx_cycle_mj,
                "bridge_class_addon_ma": r.bridge_lorawan.delta_i_class_b_ma,
                "bridge_toa_ms": r.bridge_lorawan.profile.states[NodeState.TX][PowerCircuit.RADIO].duration_ms,
                "p2p_margin_db": r.p2p_margin_db,
                "bridge_uplink_margin_db": r.bridge_lorawan.adr_info.link_margin_db,
                "bridge_rx1_margin_db": r.bridge_lorawan.downlink_rx1_margin_db,
                "bridge_rx2_margin_db": r.bridge_lorawan.downlink_rx2_margin_db,
                "end_to_end_link_feasible": r.end_to_end_link_feasible,
                "warnings": " | ".join(r.warnings)}

    if section == "cycle":
        tables = {"bridge_sf": pd.DataFrame([
            point({"sf": sf}, bridge_sf=sf, bridge_adr_enabled=False) for sf in range(7, 13)
        ])}
        if parameters.get("measured_airtime_ms") is None:
            tables["p2p_rate"] = pd.DataFrame([
                point({"rate_bps": rate}, air_rate_bps=rate)
                for rate in config["p2p_links"]["EBYTE"]["air_rate_levels_bps"]
            ])
        return tables
    if section == "sleep":
        periods = sorted(set(np.geomspace(60, 86400, 20).tolist() + [base.tx_period_s]))
        return {"sleep_period": pd.DataFrame([
            point({"period_min": period / 60, "sleep_mode": mode}, tx_period_s=period, sleep_mode=mode)
            for mode in (deep, shallow) for period in periods
        ])}
    if section == "hardware":
        return {
            "sensor_cpu": pd.DataFrame([point({"cpu_mhz": f}, cpu_freq_mhz=f) for f in sensor_cpu]),
            "bridge_wait_cpu": pd.DataFrame([
                point({"cpu_mhz": f, "wait_mode": wait}, bridge_cpu_freq_mhz=f, bridge_wait_mode=wait)
                for wait in ("light_sleep", "active") for f in bridge_cpu
            ]),
            "bridge_class": pd.DataFrame([
                point({"class": label}, bridge_device_class=cls, bridge_ping_periodicity_k=k)
                for label, cls, k in [("A", "A", None), ("B k=7", "B", 7), ("B k=4", "B", 4), ("B k=1", "B", 1)]
            ]),
        }
    if section == "range":
        p2p_distances = sorted(set(np.geomspace(.05, max(15, parameters.get("distance_km", .5)), 28).tolist()
                                   + [parameters.get("distance_km", .5)]))
        bridge_distances = sorted(set(np.geomspace(.05, max(15, parameters.get("bridge_distance_km", .5)), 28).tolist()
                                      + [parameters.get("bridge_distance_km", .5)]))
        return {
            "p2p_distance": pd.DataFrame([point({"distance_km": d}, distance_km=d) for d in p2p_distances]),
            "bridge_adr": pd.DataFrame([
                point({"distance_km": d, "mode": label}, bridge_distance_km=d,
                      bridge_adr_enabled=adr, bridge_sf=12)
                for label, adr in [("SF12", False), ("ADR", True)] for d in bridge_distances
            ]),
        }
    if section == "factors":
        current_mode = parameters.get("sleep_mode") or deep
        current_freq = parameters.get("cpu_freq_mhz") or config["mcu_configs"][sensor_mcu]["base_freq_mhz"]
        candidates = [
            ("Период передачи", base.tx_period_s, base.tx_period_s * 2, {"tx_period_s": base.tx_period_s * 2}),
            ("Частота МК сенсора", current_freq, sensor_cpu[-1] if current_freq != sensor_cpu[-1] else sensor_cpu[0],
             {"cpu_freq_mhz": sensor_cpu[-1] if current_freq != sensor_cpu[-1] else sensor_cpu[0]}),
            ("Сон сенсора", current_mode, shallow if current_mode == deep else deep,
             {"sleep_mode": shallow if current_mode == deep else deep}),
            ("Класс моста", base.bridge_lorawan.device_class, "B" if base.bridge_lorawan.device_class == "A" else "A",
             {"bridge_device_class": "B" if base.bridge_lorawan.device_class == "A" else "A", "bridge_ping_periodicity_k": 4}),
            ("Ожидание RX моста", base.bridge_lorawan.wait_mode, "active" if base.bridge_lorawan.wait_mode == "light_sleep" else "light_sleep",
             {"bridge_wait_mode": "active" if base.bridge_lorawan.wait_mode == "light_sleep" else "light_sleep"}),
        ]
        if parameters.get("measured_airtime_ms") is None:
            limit = min(51, config["p2p_links"]["EBYTE"]["lorawan_forward_payload_limit_bytes"],
                        config["p2p_links"]["EBYTE"]["subpacket_bytes"]) - ENVELOPE_BYTES
            target = limit if base.payload_bytes != limit else 1
            candidates.append(("Данные сенсора", base.payload_bytes, target, {"payload_bytes": target}))
        rows = []
        for name, before, after, overrides in candidates:
            row = point({"factor": name, "base": str(before), "varied": str(after)}, **overrides)
            row["sensor_lifetime_change_pct"] = 100 * (row["sensor_lifetime_days"] / base.sensor_battery.lifetime_days - 1)
            row["bridge_lifetime_change_pct"] = 100 * (row["bridge_lifetime_days"] / base.bridge_battery.lifetime_days - 1)
            rows.append(row)
        # x1=период, x2=частота МК сенсора, x3=сон сенсора. Payload и rate фиксированы,
        # поэтому одно измеренное время TX не переносится на другой формат пакета.
        design = []
        for x in PFE_2_3_DESIGN_MATRIX[:, 1:4]:
            period = base.tx_period_s if x[0] < 0 else base.tx_period_s * 2
            cpu = sensor_cpu[0] if x[1] < 0 else sensor_cpu[-1]
            mode = deep if x[2] < 0 else shallow
            design.append(point({"x1": x[0], "x2": x[1], "x3": x[2],
                                 "period_s": period, "cpu_mhz": cpu, "sleep_mode": mode},
                                tx_period_s=period, cpu_freq_mhz=cpu, sleep_mode=mode))
        design_df = pd.DataFrame(design)
        coeffs = []
        if not design_df["error"].any():
            sensor_fit = process_pfe_2_3(design_df["sensor_avg_ma"].to_numpy())
            bridge_fit = process_pfe_2_3(design_df["bridge_avg_ma"].to_numpy())
            coeffs = [{"term": term, "sensor_log10_i": sensor_fit.coefficients[term],
                       "bridge_log10_i": bridge_fit.coefficients[term]} for term in PFE_TERM_NAMES]
        return {"sensitivity": pd.DataFrame(rows), "pfe_design": design_df,
                "pfe_coefficients": pd.DataFrame(coeffs, columns=["term", "sensor_log10_i", "bridge_log10_i"])}
    raise ValueError(f"Неизвестный раздел анализа P2P: {section}")


# Кодированная матрица полного факторного эксперимента 2^3 (Таблица 2.6 отчёта)
# Столбцы: [x0, x1(SF), x2(Payload), x3(Class), x1x2, x1x3, x2x3, x1x2x3]
PFE_2_3_DESIGN_MATRIX: np.ndarray = np.array(
    [
        [+1, -1, -1, -1, +1, +1, +1, -1],  # Опыт 1
        [+1, +1, -1, -1, -1, -1, +1, +1],  # Опыт 2
        [+1, -1, +1, -1, -1, +1, -1, +1],  # Опыт 3
        [+1, +1, +1, -1, +1, -1, -1, -1],  # Опыт 4
        [+1, -1, -1, +1, +1, -1, -1, +1],  # Опыт 5
        [+1, +1, -1, +1, -1, +1, -1, -1],  # Опыт 6
        [+1, -1, +1, +1, -1, -1, +1, -1],  # Опыт 7
        [+1, +1, +1, +1, +1, +1, +1, +1],  # Опыт 8
    ],
    dtype=float,
)

PFE_TERM_NAMES: tuple[str, ...] = ("b0", "b1", "b2", "b3", "b12", "b13", "b23", "b123")


@dataclass(frozen=True)
class SensitivityItem:
    """Показатель чувствительности (эластичности) срока службы к варьированию фактора (формула 2.11)."""
    factor_name: str
    is_qualitative: bool
    base_value: Any
    varied_value: Any
    base_lifetime_years: float
    varied_lifetime_years: float
    rel_lifetime_change_pct: float  # (Delta_T / T_0) * 100 %
    elasticity_s: float             # Для количественных: (Delta_T / T_0) / (Delta_x / x_0)


@dataclass(frozen=True)
class PfeRegressionResult:
    """
    Результаты обработки полного факторного эксперимента 2^3 (Таблица 2.6, формула 2.12).
    Статистические поля равны None, если повторных наблюдений нет (расчёт по модели):
    значимость коэффициентов и однородность дисперсий проверяются только по измерениям.
    """
    y_means_log10: np.ndarray         # Средние значения отклика y_u = lg(I_avg) в 8 точках плана
    coefficients: Dict[str, float]    # Коэффициенты b0, b1, b2, b3, b12, b13, b23, b123
    n_repeats: int                    # Число повторных наблюдений n в каждой точке плана
    reproducibility_std_sy: float | None    # СКО воспроизводимости s_y
    coeff_std_error_sb: float | None        # СКО коэффициентов s_b = s_y / sqrt(N * n)
    t_critical: float | None                # Критическое значение Стьюдента t_(0.05; N*(n-1))
    significant_terms: Dict[str, bool] | None
    cochran_g_obs: float | None             # Наблюдаемое значение критерия Кохрена G_obs
    cochran_g_critical: float | None        # Критическое значение G_(0.05; n-1; N)
    cochran_dispersions_uniform: bool | None


def calc_elasticity_formula_2_11(
    t_base: float,
    t_varied: float,
    x_base: float | None = None,
    x_varied: float | None = None,
) -> float:
    """
    Формула (2.11): коэффициент чувствительности (эластичность) срока службы относительно базовой точки:
    S_x = ((T - T_0) / T_0) / ((x - x_0) / x_0).
    Для качественных факторов (когда x_base is None) возвращает относительное изменение (T - T_0) / T_0.
    """
    if t_base <= 0:
        raise ValueError("Базовый срок службы t_base должен быть положительным.")
    rel_t = (t_varied - t_base) / t_base
    if x_base is None or x_varied is None:
        return rel_t
    if abs(x_base) < 1e-15 or abs(x_varied - x_base) < 1e-15:
        raise ValueError("Некорректное приращение количественного фактора x.")
    rel_x = (x_varied - x_base) / x_base
    return rel_t / rel_x


def compute_sensitivity_ranking(
    config: Dict[str, Any],
    tx_period_s: float = 60.0,
    evaluation_overrides: Dict[str, Any] | None = None,
) -> List[SensitivityItem]:
    """
    Рассчитывает чувствительность срока службы ко всем основным факторам при заданном периоде T_tx (п. 2.8.1).
    Используется для построения диаграмм торнадо / чувствительности на коротких (1–5 мин)
    и длинных (60–1440 мин) периодах передачи для проверки гипотезы Г2.
    """
    base_kwargs = dict(evaluation_overrides or {})
    base_res = evaluate_node_energy_budget(config, tx_period_s=tx_period_s, **base_kwargs)
    t0 = base_res.battery.lifetime_years
    items: List[SensitivityItem] = []

    # 1. Количественные факторы: SF (9 -> 12), нагрузка (12 -> 51 байт), частота опроса (1/60 -> 1 Гц), частота МК (80 -> 240 МГц)
    quant_variations = [
        ("SF", float(base_res.sf), 12.0, {"sf": 12}),
        ("payload_bytes", float(base_res.payload_bytes), 51.0, {"payload_bytes": 51}),
        ("sensor_freq_hz", float(base_res.sensor_freq_hz), 1.0, {"sensor_freq_hz": 1.0}),
        ("cpu_freq_mhz", float(base_res.cpu_freq_mhz),
         float(max(config["mcu_configs"][base_res.mcu_config]["active_current_by_freq_ma"])),
         {"cpu_freq_mhz": max(config["mcu_configs"][base_res.mcu_config]["active_current_by_freq_ma"])}),
    ]

    for name, x0, x1, kwargs in quant_variations:
        if x0 == x1:
            continue
        merged_kwargs = {**base_kwargs, **kwargs}
        var_res = evaluate_node_energy_budget(config, tx_period_s=tx_period_s, **merged_kwargs)
        t1 = var_res.battery.lifetime_years
        el = calc_elasticity_formula_2_11(t0, t1, x0, x1)
        items.append(
            SensitivityItem(
                factor_name=name,
                is_qualitative=False,
                base_value=x0,
                varied_value=x1,
                base_lifetime_years=t0,
                varied_lifetime_years=t1,
                rel_lifetime_change_pct=100.0 * (t1 - t0) / t0,
                elasticity_s=el,
            )
        )

    # 2. Качественные факторы: режим сна (deep_sleep -> light_sleep), ожидание (light_sleep -> active),
    # класс устройства (A -> B, k=4), саморазряд (ideal -> li_socl2_aa)
    qual_variations = [
        ("sleep_mode", base_res.sleep_mode,
         "light_sleep" if base_res.mcu_config == "K1_ESP32" else "idle",
         {"sleep_mode": "light_sleep" if base_res.mcu_config == "K1_ESP32" else "idle"}),
        ("wait_mode", base_res.wait_mode, "active", {"wait_mode": "active"}),
        ("device_class", base_res.device_class, "B(k=4)", {"device_class": "B", "ping_periodicity_k": 4}),
        ("battery_type", base_res.battery_type, "li_socl2_aa", {"battery_type": "li_socl2_aa"}),
    ]

    for name, v0, v1, kwargs in qual_variations:
        merged_kwargs = {**base_kwargs, **kwargs}
        var_res = evaluate_node_energy_budget(config, tx_period_s=tx_period_s, **merged_kwargs)
        t1 = var_res.battery.lifetime_years
        rel = calc_elasticity_formula_2_11(t0, t1)
        items.append(
            SensitivityItem(
                factor_name=name,
                is_qualitative=True,
                base_value=v0,
                varied_value=v1,
                base_lifetime_years=t0,
                varied_lifetime_years=t1,
                rel_lifetime_change_pct=100.0 * rel,
                elasticity_s=rel,
            )
        )

    # Сортируем по убыванию абсолютного относительного влияния на срок службы
    items.sort(key=lambda it: abs(it.rel_lifetime_change_pct), reverse=True)
    return items


def cochran_critical_value(n_runs: int, n_repeats: int, alpha: float = 0.05) -> float:
    """
    Критическое значение критерия Кохрена G_(alpha; f = n - 1; N) через F-распределение:
    G = F / (F + N - 1), где F = F_(1 - alpha/N; f; (N - 1) f).
    Для N = 8, n = 5 даёт 0,3910 (n = 3 -> 0,5157).
    """
    f = n_repeats - 1
    if f < 1 or n_runs < 2:
        raise ValueError("Для критерия Кохрена нужно не менее двух опытов и двух повторений в каждом.")
    f_crit = float(stats.f.ppf(1.0 - alpha / n_runs, f, (n_runs - 1) * f))
    return f_crit / (f_crit + n_runs - 1)


def calc_pfe_model_points(
    config: Dict[str, Any],
    tx_period_s: float = 60.0,
    evaluation_overrides: Dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Средний ток I_avg (мА) по модели в 8 точках плана Таблицы 2.6 (п. 2.8.3):
    - x1: SF (-1 -> SF7; +1 -> SF12)
    - x2: Размер полезной нагрузки (-1 -> 12 байт; +1 -> 51 байт)
    - x3: Класс устройства (-1 -> Класс A; +1 -> Класс B, k = 4)
    """
    points = np.zeros(PFE_2_3_DESIGN_MATRIX.shape[0], dtype=float)
    base_kwargs = dict(evaluation_overrides or {})
    for u, row in enumerate(PFE_2_3_DESIGN_MATRIX):
        x1, x2, x3 = row[1], row[2], row[3]
        res = evaluate_node_energy_budget(
            config,
            **base_kwargs,
            sf=7 if x1 < 0 else 12,
            payload_bytes=12 if x2 < 0 else 51,
            device_class="A" if x3 < 0 else "B",
            ping_periodicity_k=None if x3 < 0 else 4,
            tx_period_s=tx_period_s,
        )
        points[u] = res.i_avg_total_ma
    return points


def process_pfe_2_3(i_avg_ma: Sequence[float] | np.ndarray, alpha: float = 0.05) -> PfeRegressionResult:
    """
    Обработка ПФЭ 2^3 по формуле (2.12) для отклика y = lg(I_avg).
    i_avg_ma: массив формы (8,) — одно значение на опыт (модель), или (8, n) — n повторных измерений.
    При n >= 2 проверяются однородность дисперсий (Кохрен) и значимость коэффициентов (Стьюдент).
    """
    i_arr = np.asarray(i_avg_ma, dtype=float)
    if i_arr.ndim == 1:
        i_arr = i_arr[:, None]
    n_runs, n_repeats = i_arr.shape
    if n_runs != PFE_2_3_DESIGN_MATRIX.shape[0]:
        raise ValueError(f"Ожидается 8 опытов плана Таблицы 2.6, получено: {n_runs}")
    if np.any(i_arr <= 0):
        raise ValueError("Значения среднего тока должны быть положительными.")

    y_repeats = np.log10(i_arr)
    y_means = np.mean(y_repeats, axis=1)

    # Расчёт коэффициентов по формуле (2.12): b_j = (1 / N) * sum(x_uj * y_u)
    coeffs: Dict[str, float] = {
        term_name: float(np.dot(PFE_2_3_DESIGN_MATRIX[:, j], y_means) / n_runs)
        for j, term_name in enumerate(PFE_TERM_NAMES)
    }

    if n_repeats < 2:
        return PfeRegressionResult(
            y_means_log10=y_means,
            coefficients=coeffs,
            n_repeats=n_repeats,
            reproducibility_std_sy=None,
            coeff_std_error_sb=None,
            t_critical=None,
            significant_terms=None,
            cochran_g_obs=None,
            cochran_g_critical=None,
            cochran_dispersions_uniform=None,
        )

    row_variances = np.var(y_repeats, axis=1, ddof=1)

    # Проверка однородности дисперсий по критерию Кохрена
    sum_var = float(np.sum(row_variances))
    g_obs = float(np.max(row_variances) / sum_var) if sum_var > 0 else 0.0
    g_crit = cochran_critical_value(n_runs, n_repeats, alpha)

    # СКО воспроизводимости s_y и СКО коэффициентов s_b
    s_y2 = float(np.mean(row_variances))
    s_y = math.sqrt(s_y2)
    s_b = math.sqrt(s_y2 / (n_runs * n_repeats))
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n_runs * (n_repeats - 1)))

    significant = {
        term_name: bool(abs(b_j) > t_crit * s_b) for term_name, b_j in coeffs.items()
    }

    return PfeRegressionResult(
        y_means_log10=y_means,
        coefficients=coeffs,
        n_repeats=n_repeats,
        reproducibility_std_sy=s_y,
        coeff_std_error_sb=s_b,
        t_critical=t_crit,
        significant_terms=significant,
        cochran_g_obs=g_obs,
        cochran_g_critical=g_crit,
        cochran_dispersions_uniform=g_obs <= g_crit,
    )


def run_pfe_2_3_experiment(
    config: Dict[str, Any],
    tx_period_s: float = 60.0,
    measured_i_avg_ma: Sequence[Sequence[float]] | np.ndarray | None = None,
    evaluation_overrides: Dict[str, Any] | None = None,
) -> PfeRegressionResult:
    """
    Полный факторный эксперимент 2^3 по матрице Таблицы 2.6 (п. 2.8.3, формула 2.12).
    Без measured_i_avg_ma коэффициенты считаются по модели (без проверки значимости);
    с measured_i_avg_ma формы (8, n) — по измеренным средним токам серий С1, С2, С6.
    """
    if measured_i_avg_ma is not None:
        return process_pfe_2_3(measured_i_avg_ma)
    return process_pfe_2_3(calc_pfe_model_points(config, tx_period_s, evaluation_overrides))


def run_factorial_sweep(
    config: Dict[str, Any],
    sf_levels: Sequence[int] | None = None,
    adr_levels: Sequence[bool] | None = None,
    distance_km_levels: Sequence[float] | None = None,
    payload_bytes_levels: Sequence[int] | None = None,
    sensor_freq_hz_levels: Sequence[float] | None = None,
    mcu_configs_and_freqs: Sequence[tuple[str, float]] | None = None,
    class_levels: Sequence[Dict[str, Any]] | None = None,
    battery_levels: Sequence[str] | None = None,
    tx_period_min_levels: Sequence[float] | None = None,
    wait_mode_levels: Sequence[str] | None = None,
    drop_infeasible: bool = False,
    output_csv_path: str | Path | None = None,
    radio_module: str | None = None,
    band_plan: str | None = None,
    receiver_sensitivity_dbm: float | None = None,
    gateway_sensitivity_dbm: float | None = None,
) -> pd.DataFrame:
    """
    Пункт 2.8.1: выполняет полный факторный перебор заданных уровней Таблицы 2.1
    и возвращает таблицу pandas.DataFrame (при указании output_csv_path сохраняет в CSV).
    wait_mode_levels: режимы ожидания окон RX1/RX2 для гипотезы Г5 (по умолчанию — из базовой точки).
    drop_infeasible: исключить сочетания, при которых связь со шлюзом невозможна (link_feasible = False).
    """
    fac = config["experiment_grid"]["factors"]
    sfs = sf_levels if sf_levels is not None else fac["sf_levels"]
    adrs = adr_levels if adr_levels is not None else fac["adr_levels"]
    dists = distance_km_levels if distance_km_levels is not None else fac["distance_km_levels"]
    pls = payload_bytes_levels if payload_bytes_levels is not None else fac["payload_bytes_levels"]
    f_sens = sensor_freq_hz_levels if sensor_freq_hz_levels is not None else fac["sensor_freq_hz_levels"]
    classes = class_levels if class_levels is not None else fac["class_levels"]
    bats = battery_levels if battery_levels is not None else fac["battery_levels"]
    t_mins = tx_period_min_levels if tx_period_min_levels is not None else fac["tx_period_min_levels"]
    if wait_mode_levels is not None:
        waits = wait_mode_levels
    else:
        waits = fac.get("wait_mode_levels", [config["experiment_grid"]["baseline"].get("wait_mode", "light_sleep")])

    if mcu_configs_and_freqs is None:
        mcu_pairs: List[tuple[str, float]] = []
        for m_key, freq_list in fac["mcu_freq_levels"].items():
            for f_val in freq_list:
                mcu_pairs.append((str(m_key), float(f_val)))
    else:
        mcu_pairs = list(mcu_configs_and_freqs)

    rows: List[Dict[str, Any]] = []
    for sf, adr, dist, pl, fs, (m_key, f_cpu), cls_item, bat, t_min, wait in itertools.product(
        sfs, adrs, dists, pls, f_sens, mcu_pairs, classes, bats, t_mins, waits
    ):
        res = evaluate_node_energy_budget(
            config,
            sf=int(sf),
            adr_enabled=bool(adr),
            distance_km=float(dist),
            payload_bytes=int(pl),
            sensor_freq_hz=float(fs),
            mcu_config=m_key,
            cpu_freq_mhz=f_cpu,
            device_class=str(cls_item["class"]),
            ping_periodicity_k=cls_item.get("k"),
            battery_type=str(bat),
            tx_period_s=float(t_min) * 60.0,
            wait_mode=str(wait),
            radio_module=radio_module,
            band_plan=band_plan,
            receiver_sensitivity_dbm=receiver_sensitivity_dbm,
            gateway_sensitivity_dbm=gateway_sensitivity_dbm,
        )
        if drop_infeasible and not res.link_feasible:
            continue
        rows.append(
            {
                "model_type": config.get("model_type", "unknown"),
                "radio_module": res.radio_module,
                "band_plan": res.band_plan,
                "receiver_sensitivity_dbm": res.receiver_sensitivity_dbm,
                "gateway_sensitivity_dbm": res.adr_info.receiver_sensitivity_dbm,
                "sf_input": sf,
                "sf_actual": res.sf,
                "adr_enabled": adr,
                "distance_km": dist,
                "payload_bytes": pl,
                "sensor_freq_hz": fs,
                "mcu_config": m_key,
                "cpu_freq_mhz": f_cpu,
                "device_class": res.device_class,
                "ping_k": res.ping_periodicity_k,
                "battery_type": bat,
                "tx_period_min": t_min,
                "wait_mode": res.wait_mode,
                "i_avg_ma": res.i_avg_total_ma,
                "e_tx_rx_mj": res.e_tx_rx_cycle_mj,
                "e_cycle_total_mj": res.e_cycle_total_mj,
                "e_day_j": res.e_day_total_j,
                "share_sleep_pct": res.share_sleep_pct,
                "share_radio_pct": res.share_radio_pct,
                "share_mcu_pct": res.share_mcu_board_pct,
                "lifetime_years": res.battery.lifetime_years,
                "uplink_feasible": res.adr_info.link_feasible,
                "downlink_feasible": res.downlink_feasible,
                "link_feasible": res.link_feasible,
            }
        )

    df = pd.DataFrame(rows)
    if output_csv_path is not None:
        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8")
    return df
