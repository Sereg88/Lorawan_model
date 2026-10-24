"""
Модуль источника питания (п. 2.6.1, 2.6.3).
Расчёт теоретического срока автономной работы узла с учётом саморазряда по формуле (2.8).
"""

from dataclasses import dataclass
import math
from typing import Any, Dict


HOURS_PER_DAY = 24.0
HOURS_PER_YEAR = 365.25 * 24.0  # 8766 ч


@dataclass(frozen=True)
class BatteryLifetimeResult:
    """Результат расчёта срока службы элемента питания (Таблица 2.2, формула 2.8)."""
    battery_key: str
    battery_name: str
    capacity_mah: float
    usable_fraction_eta: float
    usable_capacity_mah: float
    self_discharge_lambda_per_hour: float
    avg_current_ma: float
    lifetime_hours: float
    lifetime_days: float
    lifetime_years: float


def calc_battery_lifetime_hours(
    avg_current_ma: float,
    capacity_mah: float = 2400.0,
    usable_fraction_eta: float = 0.85,
    self_discharge_lambda_per_hour: float = 0.0,
    constant_self_discharge_current_ma: float = 0.0,
) -> float:
    """
    Формула (2.8): расчёт срока автономной работы (в часах) с учётом саморазряда.
    - Если self_discharge_lambda_per_hour > 0 (саморазряд пропорционален оставшемуся заряду):
      T = (1 / lambda) * ln(1 + (lambda * eta * C_nom) / I_avg).
    - При lambda == 0 переходит в оценку для идеального элемента:
      T = (eta * C_nom) / (I_avg + I_sd).
    """
    if avg_current_ma <= 0:
        raise ValueError(f"Средний ток должен быть положительным, получено: {avg_current_ma}")
    if capacity_mah <= 0 or not (0.0 < usable_fraction_eta <= 1.0):
        raise ValueError("Некорректные параметры ёмкости элемента питания.")
    if self_discharge_lambda_per_hour < 0 or constant_self_discharge_current_ma < 0:
        raise ValueError("Параметры саморазряда не могут быть отрицательными.")

    usable_cap_mah = usable_fraction_eta * capacity_mah
    i_eff_ma = avg_current_ma + constant_self_discharge_current_ma

    if self_discharge_lambda_per_hour < 1e-12:
        return usable_cap_mah / i_eff_ma

    lam = self_discharge_lambda_per_hour
    return math.log(1.0 + (lam * usable_cap_mah) / i_eff_ma) / lam


def evaluate_battery_from_config(
    config: Dict[str, Any],
    battery_key: str,
    avg_current_ma: float,
) -> BatteryLifetimeResult:
    """Вычисляет показатели срока службы для выбранного типа батареи из YAML-конфига."""
    bat_cfg = config["batteries"][battery_key]
    cap_mah = float(bat_cfg["capacity_mah"])
    eta = float(bat_cfg["usable_fraction_eta"])
    lam = float(bat_cfg["self_discharge_lambda_per_hour"])

    hours = calc_battery_lifetime_hours(
        avg_current_ma=avg_current_ma,
        capacity_mah=cap_mah,
        usable_fraction_eta=eta,
        self_discharge_lambda_per_hour=lam,
    )

    return BatteryLifetimeResult(
        battery_key=battery_key,
        battery_name=str(bat_cfg.get("name", battery_key)),
        capacity_mah=cap_mah,
        usable_fraction_eta=eta,
        usable_capacity_mah=cap_mah * eta,
        self_discharge_lambda_per_hour=lam,
        avg_current_ma=avg_current_ma,
        lifetime_hours=hours,
        lifetime_days=hours / HOURS_PER_DAY,
        lifetime_years=hours / HOURS_PER_YEAR,
    )