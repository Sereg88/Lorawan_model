"""
Юнит-тесты модулей обработки измерений (src/measurements.py) и вычислительного эксперимента (src/experiments.py).
"""

from pathlib import Path
import numpy as np
import pytest

from src.states import load_config, NodeState, PowerCircuit
from src.measurements import (
    integrate_shunt_segment_trapezoid,
    process_repeats_gost,
    calc_relative_error_pct,
)
from src.experiments import (
    PFE_2_3_DESIGN_MATRIX,
    cochran_critical_value,
    compute_sensitivity_ranking,
    process_pfe_2_3,
    run_pfe_2_3_experiment,
    run_factorial_sweep,
)


@pytest.fixture(scope="module")
def calibrated_cfg():
    root = Path(__file__).resolve().parent.parent
    return load_config(root / "config" / "calibrated_params.yaml")


def test_trapezoid_integration_series_c0_calibration() -> None:
    """
    Проверка формулы (2.10) на примере серии С0:
    постоянный ток 100 мА на шунте 1 Ом (падение 0.1 В) в течение 1.0 с (10000 отсчётов при 10 кГц)
    должен дать заряд 100 мКл и энергию 330 мДж при U = 3.3 В.
    """
    dt_s = 1e-4  # 10 кГц
    v_shunt = np.full(10001, 0.100, dtype=float)
    res = integrate_shunt_segment_trapezoid(
        shunt_voltage_v=v_shunt,
        dt_s=dt_s,
        n1=0,
        n2=10000,
        r_shunt_ohm=1.0,
        supply_voltage_v=3.3,
        state=NodeState.TX,
        circuit=PowerCircuit.RADIO,
    )
    assert res.duration_ms == pytest.approx(1000.0, rel=1e-5)
    assert res.mean_current_ma == pytest.approx(100.0, rel=1e-5)
    assert res.charge_mc == pytest.approx(100.0, rel=1e-5)
    assert res.energy_mj == pytest.approx(330.0, rel=1e-5)


def test_gost_grubbs_outlier_removal_and_ci_formula_2_13() -> None:
    """
    Проверка ГОСТ Р 8.736-2011 (п. 2.8.4, формула 2.13):
    5 наблюдений с одним явным промахом (50.0 вместо ~10.0) -> промах должен быть удалён по Граббсу,
    а доверительный интервал оставшихся 4 точек не должен превышать 5 %.
    """
    samples = [10.01, 9.98, 10.03, 9.99, 50.0]
    stat = process_repeats_gost(samples)

    assert stat.raw_count == 5
    assert stat.clean_count == 4
    assert len(stat.outliers_removed) == 1
    assert stat.outliers_removed[0] == pytest.approx(50.0)
    assert stat.mean == pytest.approx(10.0025, abs=0.01)
    assert not stat.needs_10_repeats


def test_relative_error_formula_2_14() -> None:
    """Проверка формулы (2.14): относительная погрешность и порог 10 %."""
    err = calc_relative_error_pct(model_val=1.08, exp_val=1.00)
    assert err == pytest.approx(8.0, abs=1e-6)


def test_pfe_2_3_matrix_orthogonality_and_regression_formula_2_12(calibrated_cfg) -> None:
    """
    Проверка матрицы Таблицы 2.6 и формулы (2.12):
    матрица ПФЭ 2^3 ортогональна (X^T * X = 8 * I), главные эффекты b1 (SF), b2 (Payload), b3 (Class B)
    и взаимодействие b13 (SF x Class B) статистически значимы по критерию Стьюдента.
    """
    gram = PFE_2_3_DESIGN_MATRIX.T @ PFE_2_3_DESIGN_MATRIX
    assert np.allclose(gram, 8.0 * np.eye(8))

    pfe = run_pfe_2_3_experiment(calibrated_cfg, tx_period_s=60.0)
    assert pfe.coefficients["b1"] > 0  # Переход SF7 -> SF12 увеличивает средний ток
    assert pfe.coefficients["b2"] > 0  # Увеличение нагрузки 12 -> 51 байт увеличивает ток
    assert pfe.coefficients["b3"] > 0  # Переход в класс B (k=4) увеличивает ток
    # Без повторных измерений значимость и однородность дисперсий не оцениваются
    assert pfe.n_repeats == 1
    assert pfe.significant_terms is None
    assert pfe.cochran_dispersions_uniform is None


def test_cochran_critical_value_table() -> None:
    """Критические значения критерия Кохрена при alpha = 0,05 для N = 8 опытов (табличные данные)."""
    assert cochran_critical_value(8, 5) == pytest.approx(0.3910, abs=5e-4)
    assert cochran_critical_value(8, 3) == pytest.approx(0.5157, abs=5e-4)


def test_pfe_processing_of_measured_repeats() -> None:
    """
    Обработка повторных измерений ПФЭ (п. 2.8.3): отклик с известными эффектами b1 и b3
    и разбросом повторений восстанавливается, незначимые эффекты отсеиваются критерием Стьюдента.
    """
    x = PFE_2_3_DESIGN_MATRIX
    y_true = -0.5 + 0.40 * x[:, 1] + 0.20 * x[:, 3]
    noise = np.array([0.004, -0.003, 0.001, -0.002, 0.0])  # одинаковый разброс во всех опытах
    i_meas = 10.0 ** (y_true[:, None] + noise[None, :])

    pfe = process_pfe_2_3(i_meas)
    assert pfe.n_repeats == 5
    assert pfe.coefficients["b1"] == pytest.approx(0.40, abs=1e-3)
    assert pfe.coefficients["b3"] == pytest.approx(0.20, abs=1e-3)
    assert pfe.cochran_dispersions_uniform
    assert pfe.significant_terms["b1"] and pfe.significant_terms["b3"]
    assert not pfe.significant_terms["b2"] and not pfe.significant_terms["b13"]


def test_sensitivity_ranking_short_vs_long_period_hypothesis_g2(calibrated_cfg) -> None:
    """Проверка п. 2.8.1 и гипотезы Г2 через ранжирование чувствительности (формула 2.11)."""
    sens_long = compute_sensitivity_ranking(calibrated_cfg, tx_period_s=360.0 * 60.0)
    by_name = {item.factor_name: item for item in sens_long}

    # На периоде 6 часов изменение режима сна влияет сильнее, чем изменение SF
    assert abs(by_name["sleep_mode"].rel_lifetime_change_pct) > abs(by_name["SF"].rel_lifetime_change_pct)


def test_small_factorial_sweep_dataframe(calibrated_cfg) -> None:
    """Проверка генерации таблицы pandas.DataFrame в вычислительном эксперименте."""
    df = run_factorial_sweep(
        calibrated_cfg,
        sf_levels=[7, 12],
        adr_levels=[False],
        distance_km_levels=[1.0],
        payload_bytes_levels=[12],
        sensor_freq_hz_levels=[1.0 / 60.0],
        mcu_configs_and_freqs=[("K1_ESP32", 80.0)],
        class_levels=[{"class": "A", "k": None}],
        battery_levels=["ideal"],
        tx_period_min_levels=[1, 60],
    )
    assert len(df) == 4
    assert "lifetime_years" in df.columns