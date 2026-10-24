"""
Модуль обработки экспериментальных измерений и осциллограмм (п. 2.6.3, 2.7.1, 2.8.4).
Реализует:
- формулу (2.10): численное интегрирование профиля тока на шунте методом трапеций по меткам GPIO;
- статистическую обработку по ГОСТ Р 8.736-2011 (критерий Граббса, доверительные границы Стьюдента, ф. 2.13);
- формулу (2.14): сопоставление модели и эксперимента по состояниям и проверку критерия |delta| <= 10%.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence
import numpy as np
from scipy import integrate, stats

from src.states import CycleProfile, NodeState, PowerCircuit


@dataclass(frozen=True)
class TrapezoidStateResult:
    """Результат интегрирования одного состояния по осциллограмме (формула 2.10)."""
    state: NodeState
    circuit: PowerCircuit
    n1: int
    n2: int
    duration_ms: float
    mean_current_ma: float
    charge_mc: float
    energy_mj: float


@dataclass(frozen=True)
class GostStatResult:
    """Результат обработки серии повторных наблюдений по ГОСТ Р 8.736-2011 (формула 2.13)."""
    raw_count: int
    clean_count: int
    outliers_removed: List[float]
    mean: float
    std: float
    t_student: float
    half_width: float          # Доверительная граница Delta = t * s / sqrt(n)
    ci_lower: float
    ci_upper: float
    rel_half_width_pct: float  # Относительная полуширина ДИ, %
    needs_10_repeats: bool     # True, если доверительный интервал превышает 5 % среднего (п. 2.8.4)


@dataclass(frozen=True)
class ModelComparisonRow:
    """Строка сопоставления модельного и экспериментального показателя (формула 2.14)."""
    name: str
    model_val: float
    exp_val: float
    rel_error_pct: float       # delta, %
    within_10_pct: bool        # Выполняется ли критерий |delta| <= 10 %


def integrate_shunt_segment_trapezoid(
    shunt_voltage_v: Sequence[float],
    dt_s: float,
    n1: int,
    n2: int,
    r_shunt_ohm: float = 1.0,
    supply_voltage_v: float = 3.3,
    state: NodeState = NodeState.TX,
    circuit: PowerCircuit = PowerCircuit.RADIO,
) -> TrapezoidStateResult:
    """
    Формула (2.10): вычисляет заряд Q (мКл) и энергию E (мДж) состояния методом трапеций
    по отсчётам падения напряжения на измерительном шунте между индексами n1 и n2.
    """
    u_arr = np.asarray(shunt_voltage_v, dtype=float)
    if n1 < 0 or n2 >= len(u_arr) or n2 <= n1:
        raise ValueError(f"Некорректные границы сегмента осциллограммы: n1={n1}, n2={n2}, len={len(u_arr)}")
    if dt_s <= 0 or r_shunt_ohm <= 0:
        raise ValueError("Шаг дискретизации dt_s и сопротивление шунта r_shunt_ohm должны быть положительными.")

    segment_v = u_arr[n1 : n2 + 1]
    # Переводим напряжение на шунте (В) в ток (мА): I(мА) = (U_sh(В) / R_sh(Ом)) * 1000
    segment_current_ma = (segment_v / r_shunt_ohm) * 1000.0

    # Интеграл мА по секундам сразу даёт заряд в милликулонах (мКл)
    charge_mc = float(integrate.trapezoid(segment_current_ma, dx=dt_s))
    duration_s = (n2 - n1) * dt_s
    duration_ms = duration_s * 1000.0
    mean_current_ma = charge_mc / duration_s
    energy_mj = supply_voltage_v * charge_mc

    return TrapezoidStateResult(
        state=state,
        circuit=circuit,
        n1=n1,
        n2=n2,
        duration_ms=duration_ms,
        mean_current_ma=mean_current_ma,
        charge_mc=charge_mc,
        energy_mj=energy_mj,
    )


def grubbs_critical_value(n: int, alpha: float = 0.05) -> float:
    """Критическое значение критерия Граббса для двустороннего отсева грубых погрешностей (ГОСТ Р 8.736-2011)."""
    if n < 3:
        return float("inf")
    t_crit = float(stats.t.ppf(1.0 - alpha / (2.0 * n), df=n - 2))
    return ((n - 1) / math.sqrt(n)) * math.sqrt((t_crit ** 2) / (n - 2 + t_crit ** 2))


def process_repeats_gost(
    values: Sequence[float],
    confidence: float = 0.95,
    alpha_grubbs: float = 0.05,
    max_rel_ci_pct: float = 5.0,
) -> GostStatResult:
    """
    Обрабатывает серию повторных измерений (по умолчанию n = 5) согласно ГОСТ Р 8.736-2011 (п. 2.8.4):
    1) Исключает грубые промахи по критерию Граббса.
    2) Вычисляет выборочное среднее, СКО и доверительный интервал по формуле (2.13).
    3) Проверяет, превышает ли полуширина доверительного интервала 5 % от среднего.
    """
    clean = [float(x) for x in values]
    if len(clean) < 2:
        raise ValueError("Для статистической обработки требуется минимум 2 повторных наблюдения.")

    raw_count = len(clean)
    outliers: List[float] = []

    while len(clean) >= 3:
        arr = np.asarray(clean, dtype=float)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1))
        if std_val < 1e-15:
            break
        deviations = np.abs(arr - mean_val)
        max_idx = int(np.argmax(deviations))
        g_obs = float(deviations[max_idx] / std_val)
        g_crit = grubbs_critical_value(len(clean), alpha=alpha_grubbs)
        if g_obs > g_crit:
            outliers.append(clean.pop(max_idx))
        else:
            break

    arr_final = np.asarray(clean, dtype=float)
    n_clean = len(arr_final)
    mean_final = float(np.mean(arr_final))
    std_final = float(np.std(arr_final, ddof=1)) if n_clean > 1 else 0.0

    alpha_ci = 1.0 - confidence
    t_val = float(stats.t.ppf(1.0 - alpha_ci / 2.0, df=max(1, n_clean - 1)))
    half_width = t_val * std_final / math.sqrt(n_clean)
    rel_pct = (100.0 * half_width / abs(mean_final)) if abs(mean_final) > 1e-15 else 0.0

    return GostStatResult(
        raw_count=raw_count,
        clean_count=n_clean,
        outliers_removed=outliers,
        mean=mean_final,
        std=std_final,
        t_student=t_val,
        half_width=half_width,
        ci_lower=mean_final - half_width,
        ci_upper=mean_final + half_width,
        rel_half_width_pct=rel_pct,
        needs_10_repeats=(rel_pct > max_rel_ci_pct),
    )


def calc_relative_error_pct(model_val: float, exp_val: float) -> float:
    """
    Формула (2.14): относительная погрешность модели
    delta = ((X_мод - X_эксп) / X_эксп) * 100 %.
    """
    if abs(exp_val) < 1e-15:
        raise ValueError("Экспериментальное значение равно нулю, расчёт относительной погрешности невозможен.")
    return 100.0 * (model_val - exp_val) / exp_val


def compare_profiles_by_state(
    model_profile: CycleProfile,
    exp_profile: CycleProfile,
    circuit: PowerCircuit | None = None,
    tolerance_pct: float = 10.0,
) -> Dict[NodeState, ModelComparisonRow]:
    """
    Пофазовое сопоставление зарядов состояний между моделью и экспериментом (формула 2.14).
    Если circuit=None, сравнивается суммарный заряд обеих цепей питания.
    """
    results: Dict[NodeState, ModelComparisonRow] = {}
    common_states = [st for st in model_profile.states if st in exp_profile.states]

    for st in common_states:
        q_mod = model_profile.get_state_charge_mc(st, circuit=circuit)
        q_exp = exp_profile.get_state_charge_mc(st, circuit=circuit)
        if q_exp > 1e-12:
            err = calc_relative_error_pct(q_mod, q_exp)
            results[st] = ModelComparisonRow(
                name=f"{st.value}_charge_mc",
                model_val=q_mod,
                exp_val=q_exp,
                rel_error_pct=err,
                within_10_pct=(abs(err) <= tolerance_pct),
            )
    return results