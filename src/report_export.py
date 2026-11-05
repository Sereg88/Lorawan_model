"""
Модуль автоматической выгрузки графиков (300 dpi PNG и векторный PDF) и таблиц (CSV)
для вставки в отчёт НИР / магистерскую диссертацию.
"""

from pathlib import Path
from typing import Any, Dict, List
from io import BytesIO
import json
import zipfile
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import yaml

from src.states import CLASS_A_ACTIVE_SEQUENCE, PowerCircuit
from src.phy import calc_symbol_duration_ms, generate_table_2_4
from src.node import evaluate_node_energy_budget
from src.experiments import (
    compute_p2p_analysis, compute_sensitivity_ranking, p2p_result_tables, run_pfe_2_3_experiment,
)
from src.p2p import evaluate_p2p_bridge


PLOT_SIZE = (6.2, 4.2)
PLOT_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd")
P2P_PLOT_LABELS = {
    "radio_mj": "Радиомодуль", "mcu_sensor_mj": "МК и датчик",
    "sensor_avg_ma": "Сенсор", "bridge_avg_ma": "Мост",
    "sensor_lifetime_days": "Сенсор", "bridge_lifetime_days": "Мост",
    "sensor_lifetime_change_pct": "Сенсор", "bridge_lifetime_change_pct": "Мост",
    "sensor_log10_i": "Сенсор", "bridge_log10_i": "Мост",
    "deep_sleep": "Глубокий сон", "power_down": "Power-down",
    "light_sleep": "Light sleep / Idle", "idle": "Idle", "active": "Активное ожидание",
    "bridge_toa_ms": "Время TX", "bridge_transaction_mj": "Энергия транзакции",
    "p2p_margin_db": "Запас P2P", "energy_mj": "Энергия",
    "ADR": "ADR вкл.", "SF12": "SF12 (без ADR)",
    "LoRaWAN + MCU": "LoRaWAN + МК", "E220 continuous RX": "E220: непрерывный RX",
    "Bridge processing above sleep": "Обработка контейнера",
}


def make_analysis_figure(
    data: pd.DataFrame, *, title: str, xlabel: str, ylabel: str,
    kind: str = "line", stacked: bool = False, log_x: bool = False,
    colors: tuple[str, ...] | None = None, secondary_y: str | None = None,
    secondary_ylabel: str | None = None,
) -> Figure:
    """Единое оформление Matplotlib для интерфейса E220 и его PNG/PDF-отчёта."""
    fig = Figure(figsize=PLOT_SIZE)
    ax = fig.subplots()
    palette = colors or PLOT_COLORS
    ax.set_prop_cycle(color=palette)
    labels = [P2P_PLOT_LABELS.get(str(c), str(c)) for c in data.columns]
    x = np.arange(len(data))
    for i, column in enumerate(data):
        values = data[column].to_numpy(dtype=float)
        color = palette[i % len(palette)]
        if colors is None:
            color = {"ADR вкл.": "#1f77b4", "SF12 (без ADR)": "#d62728"}.get(labels[i].split(":")[0], color)
        if kind in ("bar", "barh"):
            width = .72 if stacked else .72 / max(1, len(data.columns))
            offset = 0 if stacked else (i - (len(data.columns) - 1) / 2) * width
            bottom = data.iloc[:, :i].sum(axis=1).to_numpy() if stacked else np.zeros(len(data))
            bar_colors = ["#d62728" if value < 0 else "#2ca02c" for value in values] if kind == "barh" and len(data.columns) == 1 else color
            if kind == "barh":
                ax.barh(x + offset, values, height=width, left=bottom, color=bar_colors, label=labels[i])
            else:
                ax.bar(x + offset, values, width=width, bottom=bottom, color=bar_colors, label=labels[i])
        else:
            plot_ax = ax.twinx() if column == secondary_y else ax
            if column == secondary_y:
                plot_ax.set_ylabel(secondary_ylabel or labels[i], color=color)
                plot_ax.tick_params(axis="y", labelcolor=color)
            if kind == "step":
                plot_ax.step(data.index, values, where="mid", color=color, linewidth=2, label=labels[i])
            else:
                plot_ax.plot(data.index, values, marker="o", markersize=3.5,
                             linestyle="--" if i else "-", color=color, linewidth=2, label=labels[i])
    if kind in ("bar", "barh"):
        tick_labels = [P2P_PLOT_LABELS.get(str(v), str(v)) for v in data.index]
        if kind == "barh":
            ax.set_yticks(x, tick_labels)
            ax.axvline(0, color="black", linewidth=.8)
        else:
            ax.set_xticks(x, tick_labels, rotation=12 if max(map(len, tick_labels), default=0) > 12 else 0)
    if log_x:
        ax.set_xscale("log")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if secondary_y:
        ax.set_ylabel(ylabel, color=palette[0])
        ax.tick_params(axis="y", labelcolor=palette[0])
    ax.grid(axis="x" if kind == "barh" else "y" if kind == "bar" else "both",
            which="both" if log_x else "major", linestyle="--", alpha=.5)
    ax.set_axisbelow(True)
    lines, legend_labels = [], []
    for axis in fig.axes:
        handles, names = axis.get_legend_handles_labels()
        lines.extend(handles)
        legend_labels.extend(names)
        axis.tick_params(labelsize=9)
    if len(data.columns) > 1 or secondary_y:
        ax.legend(lines, legend_labels, fontsize=8.5)
    fig.tight_layout()
    return fig


def export_p2p_report_bundle(config: dict, parameters: dict, comparison_configs: dict | None = None) -> bytes:
    """Воспроизводимый отчёт P2P в памяти: параметры, исходный YAML, CSV, PNG/PDF."""
    result = evaluate_p2p_bridge(config, **parameters)
    tables = p2p_result_tables(result)
    for section in ("cycle", "sleep", "hardware", "range", "factors"):
        tables.update(compute_p2p_analysis(config, parameters, section))
    comparison = []
    for name, cfg in (comparison_configs or {}).items():
        try:
            row = p2p_result_tables(evaluate_p2p_bridge(cfg, **parameters))["summary"].iloc[0].to_dict()
            comparison.append({"model": name, "data_status": cfg.get("data_status", "unspecified"), **row})
        except ValueError as err:
            comparison.append({"model": name, "error": str(err)})
    if comparison:
        tables["model_comparison"] = pd.DataFrame(comparison)
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("parameters.json", json.dumps(parameters, ensure_ascii=False, indent=2))
        bundle.writestr("model_config.yaml", yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        for name, cfg in (comparison_configs or {}).items():
            # Имена фиксированы приложением: apriori/calibrated.
            if name in ("apriori", "calibrated"):
                bundle.writestr(f"{name}_config.yaml", yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        for name, frame in tables.items():
            bundle.writestr(f"tables/{name}.csv", frame.to_csv(index=False).encode("utf-8-sig"))
        bundle.writestr("README.txt", (
            "E220 P2P → прикладной мост → LoRaWAN. Аналитический расчёт, не измерение.\n"
            f"data_status: {config.get('data_status', 'unspecified')}. "
            "placeholder означает условные значения до калибровки.\n"
            f"TX E220: {result.airtime_source}, {result.airtime_ms:.3f} мс.\n"
            "Токи приведены к опорной шине; батареи сенсора и моста считаются отдельно.\n"
            "CSV содержит ошибки недопустимых точек; они пропущены на графиках.\n"
            "ПФЭ: x1=период (T, 2T), x2=частота МК сенсора (min, max), "
            "x3=сон сенсора (deep/power_down, light/idle); y=lg(I_avg, мА).\n"
            "Критерии значимости без повторных измерений не рассчитываются.\n\n"
            + "\n".join(result.warnings)
        ).encode("utf-8"))

        def save_plot(name, frame, x, ys, xlabel, ylabel, *, group=None, bar=False, log_x=False,
                      stacked=False, colors=None):
            if frame.empty or any(c not in frame for c in [x, *ys]):
                return
            if group:
                chart = frame.pivot(index=x, columns=group, values=ys)
                chart.columns = [f"{P2P_PLOT_LABELS.get(str(label), label)}: {P2P_PLOT_LABELS.get(str(y), y)}"
                                 for y, label in chart.columns]
            else:
                chart = frame.set_index(x)[ys]
            fig = make_analysis_figure(chart, title="E220: " + ylabel, xlabel=xlabel, ylabel=ylabel,
                                       kind="bar" if bar else "line", log_x=log_x, stacked=stacked, colors=colors)
            try:
                for extension in ("png", "pdf"):
                    buffer = BytesIO()
                    fig.savefig(buffer, format=extension, dpi=300, bbox_inches="tight")
                    bundle.writestr(f"figures/{name}.{extension}", buffer.getvalue())
            finally:
                fig.clear()

        save_plot("sensor_energy", tables["sensor_phases"], "state", ["radio_mj", "mcu_sensor_mj"],
                  "Состояние сенсора", "Энергия, мДж", bar=True, stacked=True)
        save_plot("bridge_energy", tables["bridge_energy"], "component", ["energy_mj"],
                  "Составляющая моста", "Энергия, мДж", bar=True)
        save_plot("sleep_period", tables["sleep_period"], "period_min",
                  ["sensor_lifetime_days", "bridge_lifetime_days"], "Период, мин", "Срок работы, сут.",
                  group="sleep_mode", log_x=True)
        save_plot("bridge_sf", tables["bridge_sf"], "sf", ["bridge_transaction_mj"],
                  "SF моста", "Энергия транзакции LoRaWAN, мДж")
        save_plot("bridge_wait_cpu", tables["bridge_wait_cpu"], "cpu_mhz", ["bridge_avg_ma"],
                  "Частота МК моста, МГц", "Средний ток моста, мА", group="wait_mode")
        class_chart = tables["bridge_class"].copy()
        class_chart["Базовый расход класса A"] = class_chart["bridge_avg_ma"] - class_chart["bridge_class_addon_ma"]
        class_chart["Надбавка класса B"] = class_chart["bridge_class_addon_ma"]
        save_plot("bridge_class", class_chart, "class", ["Базовый расход класса A", "Надбавка класса B"],
                  "Класс LoRaWAN моста", "Средний ток моста, мА", bar=True, stacked=True,
                  colors=("#2ca02c", "#d62728"))
        save_plot("p2p_distance", tables["p2p_distance"], "distance_km", ["p2p_margin_db"],
                  "Сенсор → мост, км", "Запас линии, дБ", log_x=True)
        feasible_adr = tables["bridge_adr"].copy()
        feasible_adr.loc[~feasible_adr["end_to_end_link_feasible"].fillna(False).astype(bool), "bridge_avg_ma"] = np.nan
        save_plot("bridge_adr", feasible_adr, "distance_km", ["bridge_avg_ma"],
                  "Мост → шлюз, км", "Средний ток моста, мА", group="mode", log_x=True)
        save_plot("sensitivity", tables["sensitivity"], "factor",
                  ["sensor_lifetime_change_pct", "bridge_lifetime_change_pct"], "Изменяемый фактор",
                  "Изменение срока работы, %", bar=True)
        save_plot("pfe", tables["pfe_coefficients"], "term", ["sensor_log10_i", "bridge_log10_i"],
                  "Коэффициент ПФЭ", "y = lg(I_avg, мА)", bar=True)
    return archive.getvalue()


def _save_fig(fig: plt.Figure, figures_dir: Path, base_name: str, dpi: int = 300) -> List[Path]:
    """Сохраняет фигуру Matplotlib одновременно в PNG (300 dpi) и векторный PDF."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{base_name}.png"
    pdf_path = figures_dir / f"{base_name}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def export_full_report_bundle(
    apriori_cfg: Dict[str, Any],
    calibrated_cfg: Dict[str, Any],
    active_cfg: Dict[str, Any],
    current_params: Dict[str, Any],
    output_dir: str | Path = "results",
) -> Path:
    """
    Генерирует полный комплект таблиц (CSV) и графиков (PNG 300 dpi + PDF) для отчёта
    и упаковывает их в единый архив `results/report_bundle.zip`.
    Возвращает путь к созданному ZIP-архиву.
    """
    radio_context = {
        key: current_params[key]
        for key in ("radio_module", "band_plan", "receiver_sensitivity_dbm", "gateway_sensitivity_dbm")
        if key in current_params
    }

    def evaluate_for_report(config, **overrides):
        return evaluate_node_energy_budget(config, **{**radio_context, **overrides})

    out_root = Path(output_dir)
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    generated_files: List[Path] = []

    # -----------------------------------------------------------------
    # 1. ТАБЛИЦА 2.4: Время передачи восходящего сообщения (ToA, мс)
    # -----------------------------------------------------------------
    t24 = generate_table_2_4()
    t24_rows = []
    for sf in [7, 8, 9, 10, 11, 12]:
        t24_rows.append({
            "SF": str(sf),
            "T_sym_ms": round(calc_symbol_duration_ms(sf), 3),
            "L_4_bytes_ms": round(t24[sf][4], 1),
            "L_12_bytes_ms": round(t24[sf][12], 1),
            "L_24_bytes_ms": round(t24[sf][24], 1),
            "L_51_bytes_ms": round(t24[sf][51], 1),
        })
    t24_rows.append({
        "SF": "SF12 / SF7",
        "T_sym_ms": 32.0,
        "L_4_bytes_ms": round(t24[12][4] / t24[7][4], 1),
        "L_12_bytes_ms": round(t24[12][12] / t24[7][12], 1),
        "L_24_bytes_ms": round(t24[12][24] / t24[7][24], 1),
        "L_51_bytes_ms": round(t24[12][51] / t24[7][51], 1),
    })
    p_t24 = tables_dir / "table_2_4_toa_ms.csv"
    pd.DataFrame(t24_rows).to_csv(p_t24, index=False, encoding="utf-8-sig")
    generated_files.append(p_t24)

    # -----------------------------------------------------------------
    # 2. ПОФАЗОВЫЙ ПРОФИЛЬ ТЕКУЩЕЙ КОНФИГУРАЦИИ И ГРАФИК БЮДЖЕТА
    # -----------------------------------------------------------------
    res_cur = evaluate_for_report(active_cfg, **current_params)
    p_link = tables_dir / "current_link_budget.csv"
    pd.DataFrame([{
        "radio_module": res_cur.radio_module,
        "band_plan": res_cur.band_plan,
        "radio_supply_voltage_v": res_cur.radio_supply_voltage_v,
        "reference_voltage_v": res_cur.profile.voltage_v,
        "node_reference_sensitivity_dbm": res_cur.receiver_sensitivity_dbm,
        "gateway_uplink_sensitivity_dbm": res_cur.adr_info.receiver_sensitivity_dbm,
        "uplink_margin_db": res_cur.adr_info.link_margin_db,
        "rx1_sensitivity_dbm": res_cur.rx1_sensitivity_dbm,
        "rx2_sensitivity_dbm": res_cur.rx2_sensitivity_dbm,
        "rx1_margin_db": res_cur.downlink_rx1_margin_db,
        "rx2_margin_db": res_cur.downlink_rx2_margin_db,
        "beacon_margin_db": res_cur.beacon_margin_db,
        "ping_slot_margin_db": res_cur.ping_slot_margin_db,
        "class_b_link_feasible": res_cur.class_b_link_feasible,
        "uplink_feasible": res_cur.adr_info.link_feasible,
        "downlink_feasible": res_cur.downlink_feasible,
        "link_feasible": res_cur.link_feasible,
        "assumptions": " | ".join(res_cur.model_warnings),
    }]).to_csv(p_link, index=False, encoding="utf-8-sig")
    generated_files.append(p_link)
    state_rows = []
    for st_enum in CLASS_A_ACTIVE_SEQUENCE:
        m_rad = res_cur.profile.states[st_enum][PowerCircuit.RADIO]
        m_mcu = res_cur.profile.states[st_enum][PowerCircuit.MCU_BOARD]
        state_rows.append({
            "Состояние": st_enum.value,
            "Длительность_мс": round(m_rad.duration_ms, 2),
            "I_радио_мА": round(m_rad.current_ma, 3),
            "I_платы_МК_мА": round(m_mcu.current_ma, 3),
            "Q_радио_мКл": round(m_rad.charge_mc, 3),
            "Q_платы_МК_мКл": round(m_mcu.charge_mc, 3),
            "E_радио_мДж": round(m_rad.energy_mj, 3),
            "E_платы_МК_мДж": round(m_mcu.energy_mj, 3),
            "E_сумм_мДж": round(m_rad.energy_mj + m_mcu.energy_mj, 3),
        })
    df_states = pd.DataFrame(state_rows)
    p_states = tables_dir / "current_cycle_states_profile.csv"
    df_states.to_csv(p_states, index=False, encoding="utf-8-sig")
    generated_files.append(p_states)

    fig_prof, ax_prof = plt.subplots(figsize=(7.0, 4.2))
    x_idx = np.arange(len(df_states))
    ax_prof.bar(x_idx, df_states["E_радио_мДж"], label="Цепь 1: Радиомодуль", color="#1f77b4")
    ax_prof.bar(
        x_idx,
        df_states["E_платы_МК_мДж"],
        bottom=df_states["E_радио_мДж"],
        label="Цепь 2: Плата МК + датчик",
        color="#ff7f0e",
    )
    ax_prof.set_xticks(x_idx)
    ax_prof.set_xticklabels(df_states["Состояние"])
    ax_prof.set_ylabel("Энергия состояния, мДж")
    ax_prof.set_title(f"Распределение энергии транзакции по состояниям (SF{res_cur.sf})")
    ax_prof.grid(axis="y", linestyle="--", alpha=0.5)
    ax_prof.legend()
    generated_files.extend(_save_fig(fig_prof, figures_dir, "fig_1_state_energy_breakdown"))

    # -----------------------------------------------------------------
    # 3. ГИПОТЕЗА Г1: Зависимость ToA и энергии транзакции от SF7..SF12
    # -----------------------------------------------------------------
    sfs = [7, 8, 9, 10, 11, 12]
    g1_rows = []
    for s in sfs:
        r_s = evaluate_for_report(
            active_cfg,
            sf=s,
            adr_enabled=False,
            payload_bytes=current_params["payload_bytes"],
            mcu_config=current_params["mcu_config"],
            cpu_freq_mhz=current_params["cpu_freq_mhz"],
            wait_mode=current_params["wait_mode"],
            radio_module=current_params["radio_module"],
        )
        toa_val = r_s.profile.states[CLASS_A_ACTIVE_SEQUENCE[1]][PowerCircuit.RADIO].duration_ms
        g1_rows.append({
            "SF": s,
            "ToA_ms": round(toa_val, 2),
            "E_tx_rx_mj": round(r_s.e_tx_rx_cycle_mj, 2),
            "Share_RX2_pct": round(r_s.share_rx2_in_tx_rx_pct, 2),
        })
    df_g1 = pd.DataFrame(g1_rows)
    p_g1 = tables_dir / "hypothesis_g1_sf_vs_energy.csv"
    df_g1.to_csv(p_g1, index=False, encoding="utf-8-sig")
    generated_files.append(p_g1)

    fig_g1, ax_g1 = plt.subplots(figsize=(7.0, 4.2))
    ax_g1_tw = ax_g1.twinx()
    l1 = ax_g1.plot(df_g1["SF"], df_g1["ToA_ms"], "o-", color="#1f77b4", linewidth=2, label="Время передачи ToA, мс")
    l2 = ax_g1_tw.plot(df_g1["SF"], df_g1["E_tx_rx_mj"], "s--", color="#d62728", linewidth=2, label="Энергия транзакции E_ц, мДж")
    ax_g1.set_xlabel("Коэффициент расширения спектра SF")
    ax_g1.set_ylabel("Время в эфире ToA, мс", color="#1f77b4")
    ax_g1_tw.set_ylabel("Энергия транзакции E_ц, мДж", color="#d62728")
    ax_g1.set_title("Гипотеза Г1: Рост времени передачи и энергии транзакции от SF")
    ax_g1.grid(True, linestyle="--", alpha=0.5)
    lines = l1 + l2
    ax_g1.legend(lines, [l.get_label() for l in lines], loc="upper left")
    generated_files.extend(_save_fig(fig_g1, figures_dir, "fig_2_hypothesis_g1_sf_energy"))

    # -----------------------------------------------------------------
    # 4. ГИПОТЕЗА Г2: Срок службы от периода передачи T_tx
    # -----------------------------------------------------------------
    periods_min = np.logspace(0, np.log10(1440), 30)
    mcu_choice = current_params["mcu_config"]
    deep_key = "deep_sleep" if mcu_choice == "K1_ESP32" else "power_down"
    shallow_key = "light_sleep" if mcu_choice == "K1_ESP32" else "idle"

    fig_g2, ax_g2 = plt.subplots(figsize=(7.0, 4.2))
    for label, kw in [
        ("SF7 + Глубокий сон", {"sf": 7, "sleep_mode": deep_key}),
        ("SF12 + Глубокий сон", {"sf": 12, "sleep_mode": deep_key}),
        ("SF7 + Неглубокий сон", {"sf": 7, "sleep_mode": shallow_key}),
    ]:
        vals = [
            evaluate_for_report(
                active_cfg,
                adr_enabled=False,
                payload_bytes=current_params["payload_bytes"],
                mcu_config=mcu_choice,
                cpu_freq_mhz=current_params["cpu_freq_mhz"],
                battery_type=current_params["battery_type"],
                tx_period_s=float(p_m) * 60.0,
                **kw,
            ).battery.lifetime_years
            for p_m in periods_min
        ]
        ax_g2.plot(periods_min, vals, linewidth=2, label=label)
    ax_g2.set_xscale("log")
    ax_g2.set_xlabel("Период передачи T_tx, мин (лог. шкала)")
    ax_g2.set_ylabel("Срок автономной работы, лет")
    ax_g2.set_title("Гипотеза Г2: Влияние режима сна и SF на срок службы")
    ax_g2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_g2.legend()
    generated_files.extend(_save_fig(fig_g2, figures_dir, "fig_3_hypothesis_g2_lifetime_vs_period"))

    # -----------------------------------------------------------------
    # 5. ГИПОТЕЗА Г3: Надбавка среднего тока в Классе B
    # -----------------------------------------------------------------
    modes_b = [
        ("Класс A", "A", None),
        ("Класс B (k=7)", "B", 7),
        ("Класс B (k=4)", "B", 4),
        ("Класс B (k=1)", "B", 1),
    ]
    i_base_list, i_addon_list = [], []
    for _, cl, k_item in modes_b:
        r_b = evaluate_for_report(
            active_cfg,
            sf=current_params["sf"],
            device_class=cl,
            ping_periodicity_k=k_item,
            tx_period_s=current_params["tx_period_s"],
        )
        i_base_list.append(r_b.i_avg_class_a_ma * 1000.0)
        i_addon_list.append(r_b.delta_i_class_b_ma * 1000.0)

    fig_g3, ax_g3 = plt.subplots(figsize=(6.8, 4.2))
    labels_b = [m[0] for m in modes_b]
    ax_g3.bar(labels_b, i_base_list, label="Базовый расход Класса A, мкА", color="#2ca02c")
    ax_g3.bar(labels_b, i_addon_list, bottom=i_base_list, label="Надбавка Класса B (маяки + ping), мкА", color="#d62728")
    ax_g3.set_ylabel("Средний ток потребления, мкА")
    ax_g3.set_title("Гипотеза Г3: Средний ток в Классе B при разной периодичности ping-слотов")
    ax_g3.legend()
    ax_g3.grid(axis="y", linestyle="--", alpha=0.5)
    generated_files.extend(_save_fig(fig_g3, figures_dir, "fig_4_hypothesis_g3_class_b"))

    # -----------------------------------------------------------------
    # 6. ГИПОТЕЗА Г4: Дальность связи и алгоритм ADR
    # -----------------------------------------------------------------
    dists = active_cfg["experiment_grid"]["factors"]["distance_km_levels"]
    g4_rows = []
    for d in dists:
        r_on = evaluate_for_report(active_cfg, adr_enabled=True, distance_km=d, payload_bytes=current_params["payload_bytes"])
        r_off = evaluate_for_report(active_cfg, adr_enabled=False, sf=12, distance_km=d, payload_bytes=current_params["payload_bytes"])
        g4_rows.append({
            "Distance_km": d,
            "SNR_dB": round(r_on.adr_info.snr_db, 2),
            "ADR_SF": r_on.sf,
            "E_ADR_on_mJ": round(r_on.e_tx_rx_cycle_mj, 2),
            "E_SF12_fixed_mJ": round(r_off.e_tx_rx_cycle_mj, 2),
        })
    df_g4 = pd.DataFrame(g4_rows)
    p_g4 = tables_dir / "hypothesis_g4_adr_vs_distance.csv"
    df_g4.to_csv(p_g4, index=False, encoding="utf-8-sig")
    generated_files.append(p_g4)

    fig_g4, (ax_g4_1, ax_g4_2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax_g4_1.plot(df_g4["Distance_km"], df_g4["E_SF12_fixed_mJ"], "r--", linewidth=2, label="ADR выкл. (SF12)")
    ax_g4_1.plot(df_g4["Distance_km"], df_g4["E_ADR_on_mJ"], "bo-", linewidth=2, label="ADR вкл.")
    ax_g4_1.set_xlabel("Расстояние до шлюза d, км")
    ax_g4_1.set_ylabel("Энергия транзакции E_ц, мДж")
    ax_g4_1.set_title("Экономия энергии передачи за счёт ADR")
    ax_g4_1.grid(True, linestyle="--", alpha=0.5)
    ax_g4_1.legend()

    ax_g4_2.step(df_g4["Distance_km"], df_g4["ADR_SF"], where="mid", color="#9467bd", linewidth=2.5, marker="o")
    ax_g4_2.set_xlabel("Расстояние до шлюза d, км")
    ax_g4_2.set_ylabel("Установившийся SF")
    ax_g4_2.set_yticks([7, 8, 9, 10, 11, 12])
    ax_g4_2.set_title("Переключение SF алгоритмом ADR от расстояния")
    ax_g4_2.grid(True, linestyle="--", alpha=0.5)
    generated_files.extend(_save_fig(fig_g4, figures_dir, "fig_5_hypothesis_g4_adr_distance"))

    # -----------------------------------------------------------------
    # 7. ГИПОТЕЗА Г5: Поведение МК в паузах приёма против тактовой частоты
    # -----------------------------------------------------------------
    freqs = [80, 160, 240] if mcu_choice == "K1_ESP32" else [2, 4, 8]
    e_light = [
        evaluate_for_report(active_cfg, mcu_config=mcu_choice, cpu_freq_mhz=f, wait_mode="light_sleep", sf=current_params["sf"]).e_tx_rx_cycle_mj
        for f in freqs
    ]
    e_active = [
        evaluate_for_report(active_cfg, mcu_config=mcu_choice, cpu_freq_mhz=f, wait_mode="active", sf=current_params["sf"]).e_tx_rx_cycle_mj
        for f in freqs
    ]
    fig_g5, ax_g5 = plt.subplots(figsize=(6.8, 4.2))
    x_f = np.arange(len(freqs))
    w = 0.35
    ax_g5.bar(x_f - w / 2, e_light, w, label="Ожидание: Light Sleep / Idle", color="#1f77b4")
    ax_g5.bar(x_f + w / 2, e_active, w, label="Ожидание: Активное (Active)", color="#ff7f0e")
    ax_g5.set_xticks(x_f)
    ax_g5.set_xticklabels([f"{f} МГц" for f in freqs])
    ax_g5.set_ylabel("Энергия транзакции E_ц, мДж")
    ax_g5.set_title("Гипотеза Г5: Режим ожидания окон приёма vs Тактовая частота МК")
    ax_g5.legend()
    ax_g5.grid(axis="y", linestyle="--", alpha=0.5)
    generated_files.extend(_save_fig(fig_g5, figures_dir, "fig_6_hypothesis_g5_wait_vs_freq"))

    # -----------------------------------------------------------------
    # 8. ГИПОТЕЗА Г6: Сравнение априорной и откалиброванной моделей
    # -----------------------------------------------------------------
    r_apr = evaluate_for_report(apriori_cfg, **current_params)
    r_cal = evaluate_for_report(calibrated_cfg, **current_params)
    df_g6 = pd.DataFrame([
        {"Показатель": "Ток сна узла, мкА", "Априорная": round(r_apr.i_sleep_total_ua, 2), "Откалиброванная": round(r_cal.i_sleep_total_ua, 2)},
        {"Показатель": "Энергия транзакции E_ц, мДж", "Априорная": round(r_apr.e_tx_rx_cycle_mj, 2), "Откалиброванная": round(r_cal.e_tx_rx_cycle_mj, 2)},
        {"Показатель": "Средний ток I_avg, мкА", "Априорная": round(r_apr.i_avg_total_ma * 1000.0, 2), "Откалиброванная": round(r_cal.i_avg_total_ma * 1000.0, 2)},
        {"Показатель": "Срок службы, лет", "Априорная": round(r_apr.battery.lifetime_years, 2), "Откалиброванная": round(r_cal.battery.lifetime_years, 2)},
    ])
    p_g6 = tables_dir / "hypothesis_g6_apriori_vs_calibrated.csv"
    df_g6.to_csv(p_g6, index=False, encoding="utf-8-sig")
    generated_files.append(p_g6)

    # -----------------------------------------------------------------
    # 9. ЧУВСТВИТЕЛЬНОСТЬ (ф. 2.11: 1 мин vs 360 мин) И ПФЭ 2^3 (Таблица 2.6)
    # -----------------------------------------------------------------
    sens_short = compute_sensitivity_ranking(active_cfg, tx_period_s=60.0, evaluation_overrides=radio_context)
    sens_long = compute_sensitivity_ranking(active_cfg, tx_period_s=360.0 * 60.0, evaluation_overrides=radio_context)

    fig_sens, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(12.5, 4.5))
    for ax_s, s_list, title_str in [
        (ax_s1, sens_short, "Короткий период (T_tx = 1 мин)"),
        (ax_s2, sens_long, "Длинный период (T_tx = 360 мин)"),
    ]:
        names = [f"{it.factor_name} ({it.base_value}→{it.varied_value})" for it in reversed(s_list)]
        vals = [it.rel_lifetime_change_pct for it in reversed(s_list)]
        cols = ["#d62728" if v < 0 else "#2ca02c" for v in vals]
        ax_s.barh(names, vals, color=cols)
        ax_s.axvline(0, color="black", linewidth=0.8)
        ax_s.set_xlabel("ΔT / T₀, %")
        ax_s.set_title(title_str)
        ax_s.grid(axis="x", linestyle="--", alpha=0.5)
    generated_files.extend(_save_fig(fig_sens, figures_dir, "fig_7_sensitivity_short_vs_long"))

    pfe = run_pfe_2_3_experiment(active_cfg, tx_period_s=60.0, evaluation_overrides=radio_context)
    df_pfe = pd.DataFrame([
        {
            "Коэффициент": k,
            "Значение_b_j": round(v, 5),
            # Значимость проверяется только по повторным измерениям (п. 2.8.3)
            "Значим_по_Стьюденту": pfe.significant_terms[k] if pfe.significant_terms is not None else None,
        }
        for k, v in pfe.coefficients.items()
    ])
    p_pfe = tables_dir / "table_2_6_pfe_2_3_regression.csv"
    df_pfe.to_csv(p_pfe, index=False, encoding="utf-8-sig")
    generated_files.append(p_pfe)

    # Упаковка всех файлов в единый ZIP-архив
    zip_path = out_root / "report_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in generated_files:
            zf.write(file_path, arcname=file_path.relative_to(out_root))

    return zip_path
