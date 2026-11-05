"""
Модуль радиоканала и алгоритма ADR (п. 2.6.1, 2.7.3).
Реализует логарифмическую модель затухания и расчёт SNR по формуле (2.6),
а также выбор установившегося значения SF и мощности передатчика по формуле (2.7).
"""

from dataclasses import dataclass
import math
from typing import Dict, Sequence, Tuple
import numpy as np


DEFAULT_SNR_THRESHOLDS_DB: Dict[int, float] = {
    7: -7.5,
    8: -10.0,
    9: -12.5,
    10: -15.0,
    11: -17.5,
    12: -20.0,
}


@dataclass(frozen=True)
class LinkBudgetResult:
    """Результаты расчёта бюджета линии связи на заданном расстоянии (формула 2.6)."""
    distance_km: float
    path_loss_db: float
    rssi_dbm: float
    noise_floor_dbm: float
    snr_db: float
    link_feasible_sf12: bool  # Выполняется ли условие SNR >= SNR_req(SF12) = -20 дБ


@dataclass(frozen=True)
class AdrConvergenceResult:
    """Результат работы алгоритма ADR (формула 2.7)."""
    distance_km: float
    adr_enabled: bool
    initial_sf: int
    final_sf: int
    initial_tx_power_dbm: float
    final_tx_power_dbm: float
    snr_db: float
    snr_margin_db: float
    n_steps: int
    link_feasible: bool
    rssi_dbm: float
    receiver_sensitivity_dbm: float
    link_margin_db: float


def calc_noise_floor_dbm(
    bw_khz: float = 125.0,
    thermal_noise_dbm_hz: float = -174.0,
    noise_figure_db: float = 6.0,
) -> float:
    """
    Вычисляет мощность теплового шума в полосе приёмника шлюза:
    N = -174 + 10 * lg(BW_Hz) + NF, дБм.
    При BW = 125 кГц и NF = 6 дБ даёт около -117.0 дБм.
    """
    bw_hz = bw_khz * 1000.0
    return thermal_noise_dbm_hz + 10.0 * math.log10(bw_hz) + noise_figure_db


def calc_link_budget(
    distance_km: float,
    tx_power_dbm: float = 10.0,
    antenna_gain_db: float = 0.0,
    ref_distance_km: float = 0.05,
    ref_loss_db: float = 70.0,
    path_loss_exponent: float = 2.8,
    bw_khz: float = 125.0,
    thermal_noise_dbm_hz: float = -174.0,
    gateway_noise_figure_db: float = 6.0,
    snr_thresholds_db: Dict[int, float] = DEFAULT_SNR_THRESHOLDS_DB,
) -> LinkBudgetResult:
    """
    Формула (2.6): логарифмическая модель затухания и отношение сигнал/шум на входе шлюза.
    Для лабораторных расстояний d < d_0 затухание также пересчитывается по логарифму (с нижней границей 1 м).
    """
    d_eff = max(distance_km, 0.001)
    path_loss_db = ref_loss_db + 10.0 * path_loss_exponent * math.log10(d_eff / ref_distance_km)
    rssi_dbm = tx_power_dbm + antenna_gain_db - path_loss_db
    noise_floor_dbm = calc_noise_floor_dbm(
        bw_khz=bw_khz,
        thermal_noise_dbm_hz=thermal_noise_dbm_hz,
        noise_figure_db=gateway_noise_figure_db,
    )
    snr_db = rssi_dbm - noise_floor_dbm
    feasible = snr_db >= snr_thresholds_db[12]

    return LinkBudgetResult(
        distance_km=distance_km,
        path_loss_db=path_loss_db,
        rssi_dbm=rssi_dbm,
        noise_floor_dbm=noise_floor_dbm,
        snr_db=snr_db,
        link_feasible_sf12=feasible,
    )


def resolve_adr(
    distance_km: float,
    adr_enabled: bool,
    fixed_sf: int = 12,
    initial_sf_when_adr: int = 12,
    tx_power_dbm: float = 10.0,
    min_tx_power_dbm: float = 2.0,
    tx_power_step_db: float = 2.0,
    antenna_gain_db: float = 0.0,
    ref_distance_km: float = 0.05,
    ref_loss_db: float = 70.0,
    path_loss_exponent: float = 2.8,
    bw_khz: float = 125.0,
    gateway_noise_figure_db: float = 6.0,
    adr_margin_db: float = 10.0,
    snr_thresholds_db: Dict[int, float] = DEFAULT_SNR_THRESHOLDS_DB,
    receiver_sensitivity_by_sf_dbm: Dict[int, float] | None = None,
) -> AdrConvergenceResult:
    """
    Формула (2.7): моделирование алгоритма ADR.
    Если adr_enabled=False, узел остаётся на fixed_sf и начальной мощности tx_power_dbm.
    Если adr_enabled=True, узел стартует с initial_sf_when_adr (по п. 2.7.3 — с SF12),
    вычисляет запас SNR и N_step = floor((SNR - SNR_req - M_adr) / 3),
    после чего снижает SF вплоть до SF7 (DR5), а оставшиеся шаги тратит на снижение мощности:
    каждый шаг увеличивает индекс TXPower на 1, т. е. снижает мощность на tx_power_step_db
    (2 дБ в планах EU433 и EU868) до min_tx_power_dbm.
    """
    lb = calc_link_budget(
        distance_km=distance_km,
        tx_power_dbm=tx_power_dbm,
        antenna_gain_db=antenna_gain_db,
        ref_distance_km=ref_distance_km,
        ref_loss_db=ref_loss_db,
        path_loss_exponent=path_loss_exponent,
        bw_khz=bw_khz,
        gateway_noise_figure_db=gateway_noise_figure_db,
        snr_thresholds_db=snr_thresholds_db,
    )

    # Это приёмник шлюза (uplink), а не чувствительность радиомодуля узла.
    thresholds = (
        {sf: lb.noise_floor_dbm + snr for sf, snr in snr_thresholds_db.items()}
        if receiver_sensitivity_by_sf_dbm is None else receiver_sensitivity_by_sf_dbm
    )
    if any(sf not in thresholds or not math.isfinite(thresholds[sf]) for sf in range(7, 13)):
        raise ValueError("Нужны конечные значения чувствительности шлюза для SF7..SF12.")
    if not math.isfinite(tx_power_step_db) or tx_power_step_db <= 0 or min_tx_power_dbm > tx_power_dbm:
        raise ValueError("Некорректные пределы мощности или шаг ADR.")

    if not adr_enabled:
        link_margin = lb.rssi_dbm - thresholds[fixed_sf]
        margin = link_margin - adr_margin_db
        return AdrConvergenceResult(
            distance_km=distance_km,
            adr_enabled=False,
            initial_sf=fixed_sf,
            final_sf=fixed_sf,
            initial_tx_power_dbm=tx_power_dbm,
            final_tx_power_dbm=tx_power_dbm,
            snr_db=lb.snr_db,
            snr_margin_db=margin,
            n_steps=0,
            link_feasible=link_margin >= 0,
            rssi_dbm=lb.rssi_dbm,
            receiver_sensitivity_dbm=thresholds[fixed_sf],
            link_margin_db=link_margin,
        )

    start_sf = initial_sf_when_adr
    margin = lb.rssi_dbm - thresholds[start_sf] - adr_margin_db
    n_steps = math.floor(margin / 3.0)

    current_sf = start_sf
    current_p_tx = tx_power_dbm
    remaining_steps = n_steps

    if remaining_steps > 0:
        # Сначала повышаем скорость (снижаем SF по 1 за каждый шаг до SF7)
        max_sf_reduction = current_sf - 7
        sf_drop = min(remaining_steps, max_sf_reduction)
        current_sf -= sf_drop
        remaining_steps -= sf_drop

        # Если после достижения SF7 остались положительные шаги, снижаем мощность передатчика (по индексу TXPower)
        if remaining_steps > 0:
            p_drop = remaining_steps * tx_power_step_db
            current_p_tx = max(min_tx_power_dbm, current_p_tx - p_drop)

    # Проверяем, выполняется ли условие связи при итоговом SF и итоговой мощности
    final_snr = lb.snr_db - (tx_power_dbm - current_p_tx)
    final_rssi = lb.rssi_dbm - (tx_power_dbm - current_p_tx)
    feasible = final_rssi >= thresholds[current_sf]

    # Если при слишком большом расстоянии запас отрицательный и даже на стартовом SF связи нет,
    # возвращаем SF12 и максимальную мощность
    if not feasible and current_sf < 12:
        # Подбираем минимальный SF, при котором выполняется условие связи SNR >= SNR_req(SF) (без запаса M_adr)
        for candidate_sf in range(7, 13):
            if lb.rssi_dbm >= thresholds[candidate_sf]:
                current_sf = candidate_sf
                current_p_tx = tx_power_dbm
                feasible = True
                break

    final_rssi = lb.rssi_dbm - (tx_power_dbm - current_p_tx)
    final_snr = final_rssi - lb.noise_floor_dbm

    return AdrConvergenceResult(
        distance_km=distance_km,
        adr_enabled=True,
        initial_sf=start_sf,
        final_sf=current_sf,
        initial_tx_power_dbm=tx_power_dbm,
        final_tx_power_dbm=current_p_tx,
        snr_db=final_snr,
        snr_margin_db=margin,
        n_steps=n_steps,
        link_feasible=feasible,
        rssi_dbm=final_rssi,
        receiver_sensitivity_dbm=thresholds[current_sf],
        link_margin_db=final_rssi - thresholds[current_sf],
    )


def fit_path_loss_params(
    distances_km: Sequence[float],
    measured_rssi_dbm: Sequence[float],
    tx_power_dbm: float = 10.0,
    antenna_gain_db: float = 0.0,
    ref_distance_km: float = 0.05,
) -> Tuple[float, float]:
    """
    Определяет параметры L_0 (затухание на опорном расстоянии d_0) и gamma (показатель затухания)
    модели (2.6) методом наименьших квадратов по результатам полевых измерений RSSI в серии С7 (п. 2.7.3).
    Возвращает кортеж (ref_loss_db, path_loss_exponent).
    """
    if len(distances_km) != len(measured_rssi_dbm) or len(distances_km) < 2:
        raise ValueError("Для аппроксимации параметров канала требуется минимум 2 точки измерений.")

    d_arr = np.asarray(distances_km, dtype=float)
    rssi_arr = np.asarray(measured_rssi_dbm, dtype=float)
    measured_loss_db = tx_power_dbm + antenna_gain_db - rssi_arr

    # X = 10 * lg(d / d_0), тогда L(d) = L_0 + gamma * X
    x = 10.0 * np.log10(d_arr / ref_distance_km)
    gamma, l0 = np.polyfit(x, measured_loss_db, 1)
    return float(l0), float(gamma)
