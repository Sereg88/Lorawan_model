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
    """Результаты обработки полного факторного эксперимента 2^3 (Таблица 2.6, формула 2.12)."""
    y_means_log10: np.ndarray         # Средние значения отклика y_u = lg(I_avg) в 8 точках плана
    coefficients: Dict[str, float]    # Коэффициенты b0, b1, b2, b3, b12, b13, b23, b123
    reproducibility_std_sy: float     # СКО воспроизводимости s_y
    coeff_std_error_sb: float         # СКО коэффициентов s_b = s_y / sqrt(N * n)
    t_critical: float                 # Критическое значение Стьюдента t_(0.05; N*(n-1))
    significant_terms: Dict[str, bool]
    cochran_g_obs: float              # Наблюдаемое значение критерия Кохрена G_obs
    cochran_dispersions_uniform: bool


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
) -> List[SensitivityItem]:
    """
    Рассчитывает чувствительность срока службы ко всем основным факторам при заданном периоде T_tx (п. 2.8.1).
    Используется для построения диаграмм торнадо / чувствительности на коротких (1–5 мин)
    и длинных (60–1440 мин) периодах передачи для проверки гипотезы Г2.
    """
    base_res = evaluate_node_energy_budget(config, tx_period_s=tx_period_s)
    t0 = base_res.battery.lifetime_years
    items: List[SensitivityItem] = []

    # 1. Количественные факторы: SF (9 -> 12), нагрузка (12 -> 51 байт), частота опроса (1/60 -> 1 Гц), частота МК (80 -> 240 МГц)
    quant_variations = [
        ("SF", float(base_res.sf), 12.0, {"sf": 12}),
        ("payload_bytes", float(base_res.payload_bytes), 51.0, {"payload_bytes": 51}),
        ("sensor_freq_hz", float(base_res.sensor_freq_hz), 1.0, {"sensor_freq_hz": 1.0}),
        ("cpu_freq_mhz", float(base_res.cpu_freq_mhz), 240.0, {"cpu_freq_mhz": 240.0}),
    ]

    for name, x0, x1, kwargs in quant_variations:
        var_res = evaluate_node_energy_budget(config, tx_period_s=tx_period_s, **kwargs)
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
        ("sleep_mode", "deep_sleep", "light_sleep", {"sleep_mode": "light_sleep"}),
        ("wait_mode", "light_sleep", "active", {"wait_mode": "active"}),
        ("device_class", "A", "B(k=4)", {"device_class": "B", "ping_periodicity_k": 4}),
        ("battery_type", "ideal", "li_socl2_aa", {"battery_type": "li_socl2_aa"}),
    ]

    for name, v0, v1, kwargs in qual_variations:
        var_res = evaluate_node_energy_budget(config, tx_period_s=tx_period_s, **kwargs)
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


def run_pfe_2_3_experiment(
    config: Dict[str, Any],
    tx_period_s: float = 60.0,
    n_repeats: int = 5,
    repeat_rel_std: float = 0.005,
    random_seed: int = 42,
) -> PfeRegressionResult:
    """
    Выполняет полный факторный эксперимент 2^3 по матрице Таблицы 2.6 (п. 2.8.3, формула 2.12):
    - x1: SF (-1 -> SF7; +1 -> SF12)
    - x2: Размер полезной нагрузки (-1 -> 12 байт; +1 -> 51 байт)
    - x3: Класс устройства (-1 -> Класс A; +1 -> Класс B, k = 4)
    Отклик: y = lg(I_avg) при T_tx = 60 с.
    """
    rng = np.random.default_rng(random_seed)
    n_runs = 8
    y_repeats = np.zeros((n_runs, n_repeats), dtype=float)

    for u in range(n_runs):
        x1, x2, x3 = PFE_2_3_DESIGN_MATRIX[u, 1], PFE_2_3_DESIGN_MATRIX[u, 2], PFE_2_3_DESIGN_MATRIX[u, 3]
        sf_val = 7 if x1 < 0 else 12
        pl_val = 12 if x2 < 0 else 51
        cls_val = "A" if x3 < 0 else "B"
        k_val = None if x3 < 0 else 4

        res = evaluate_node_energy_budget(
            config,
            sf=sf_val,
            payload_bytes=pl_val,
            device_class=cls_val,
            ping_periodicity_k=k_val,
            tx_period_s=tx_period_s,
        )
        i_nom = res.i_avg_total_ma

        # Моделируем n=5 повторных наблюдений с малой дисперсией воспроизводимости (п. 2.8.4)
        for r in range(n_repeats):
            noise_factor = 1.0 + rng.normal(0.0, repeat_rel_std) if repeat_rel_std > 0 else 1.0
            y_repeats[u, r] = math.log10(max(1e-9, i_nom * noise_factor))

    y_means = np.mean(y_repeats, axis=1)
    row_variances = np.var(y_repeats, axis=1, ddof=1) if n_repeats > 1 else np.zeros(n_runs)

    # Проверка однородности дисперсий по критерию Кохрена
    sum_var = float(np.sum(row_variances))
    g_obs = float(np.max(row_variances) / sum_var) if sum_var > 1e-18 else 0.0
    # Табличный порог Кохрена для N=8, df=4 (n=5) при alpha=0.05 равен 0.5157
    cochran_ok = g_obs <= 0.5157

    # СКО воспроизводимости s_y и СКО коэффициентов s_b
    s_y2 = float(np.mean(row_variances))
    s_y = math.sqrt(s_y2)
    s_b = math.sqrt(s_y2 / (n_runs * n_repeats)) if s_y2 > 0 else 1e-12

    df_total = n_runs * max(1, n_repeats - 1)
    t_crit = float(stats.t.ppf(1.0 - 0.05 / 2.0, df=df_total))

    # Расчёт коэффициентов по формуле (2.12): b_j = (1 / N) * sum(x_uj * y_u)
    coeffs: Dict[str, float] = {}
    significant: Dict[str, bool] = {}
    for j, term_name in enumerate(PFE_TERM_NAMES):
        b_j = float(np.dot(PFE_2_3_DESIGN_MATRIX[:, j], y_means) / n_runs)
        coeffs[term_name] = b_j
        t_j = abs(b_j) / s_b if s_b > 1e-15 else float("inf")
        significant[term_name] = bool(t_j > t_crit)

    return PfeRegressionResult(
        y_means_log10=y_means,
        coefficients=coeffs,
        reproducibility_std_sy=s_y,
        coeff_std_error_sb=s_b,
        t_critical=t_crit,
        significant_terms=significant,
        cochran_g_obs=g_obs,
        cochran_dispersions_uniform=cochran_ok,
    )


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
    output_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Пункт 2.8.1: выполняет полный факторный перебор заданных уровней Таблицы 2.1
    и возвращает таблицу pandas.DataFrame (при указании output_csv_path сохраняет в CSV).
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

    if mcu_configs_and_freqs is None:
        mcu_pairs: List[tuple[str, float]] = []
        for m_key, freq_list in fac["mcu_freq_levels"].items():
            for f_val in freq_list:
                mcu_pairs.append((str(m_key), float(f_val)))
    else:
        mcu_pairs = list(mcu_configs_and_freqs)

    rows: List[Dict[str, Any]] = []
    for sf, adr, dist, pl, fs, (m_key, f_cpu), cls_item, bat, t_min in itertools.product(
        sfs, adrs, dists, pls, f_sens, mcu_pairs, classes, bats, t_mins
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
        )
        rows.append(
            {
                "model_type": config.get("model_type", "unknown"),
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
                "i_avg_ma": res.i_avg_total_ma,
                "e_tx_rx_mj": res.e_tx_rx_cycle_mj,
                "e_cycle_total_mj": res.e_cycle_total_mj,
                "e_day_j": res.e_day_total_j,
                "share_sleep_pct": res.share_sleep_pct,
                "share_radio_pct": res.share_radio_pct,
                "share_mcu_pct": res.share_mcu_board_pct,
                "lifetime_years": res.battery.lifetime_years,
                "link_feasible": res.adr_info.link_feasible,
            }
        )

    df = pd.DataFrame(rows)
    if output_csv_path is not None:
        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8")
    return df