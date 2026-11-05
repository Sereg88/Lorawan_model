"""
Интерактивный веб-дашборд (Streamlit) для аналитической модели энергопотребления узла LoRaWAN.
Визуализирует энергетический бюджет, профили состояний, проверку гипотез Г1–Г6,
анализ чувствительности (торнадо) и результаты ПФЭ 2^3.
"""

from pathlib import Path
from html import escape
from textwrap import fill
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np
import pandas as pd
import streamlit as st

from src.states import (
    CLASS_A_ACTIVE_SEQUENCE,
    PowerCircuit,
    get_supported_band_plans,
    load_config,
    resolve_band_plan,
)
from src.phy import generate_table_2_4
from src.channel_adr import resolve_adr
from src.channel_adr import calc_noise_floor_dbm
from src.node import calc_sleep_breakeven_time_s, evaluate_node_energy_budget
from src.radio import get_radio_compatibility_note
from src.p2p import ENVELOPE_BYTES, encode_p2p_message, evaluate_p2p_bridge
from src.experiments import (
    compute_p2p_analysis, compute_sensitivity_ranking, p2p_result_tables, run_pfe_2_3_experiment,
)
from src.report_export import export_p2p_report_bundle, make_analysis_figure

# Настройка страницы
st.set_page_config(
    page_title="LoRaWAN / модель",
    page_icon='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" fill="#181818"/><rect x="4" y="4" width="56" height="56" fill="none" stroke="#b6ef95"/><text x="10" y="42" font-family="monospace" font-size="32" fill="#b6ef95">&gt;_</text></svg>',
    layout="wide",
)

ROOT_DIR = Path(__file__).resolve().parent
st.html(ROOT_DIR / "assets" / "terminal.css")


CONFIG_PATHS = {
    "apriori": ROOT_DIR / "config" / "apriori_params.yaml",
    "calibrated": ROOT_DIR / "config" / "calibrated_params.yaml",
}


@st.cache_data
def get_configs(config_mtimes: tuple[float, ...]):
    # config_mtimes входит в ключ кэша: при правке YAML-файлов конфиги перечитываются
    return {name: load_config(path) for name, path in CONFIG_PATHS.items()}


@st.cache_data(max_entries=32)
def get_p2p_analysis(config: dict, parameters: dict, section: str):
    return compute_p2p_analysis(config, parameters, section)


@st.cache_data(max_entries=4)
def get_p2p_report(config: dict, parameters: dict, comparison_configs: dict):
    return export_p2p_report_bundle(config, parameters, comparison_configs)


def show_terminal_plot(figure, *, title: str | None = None):
    """Терминальное оформление только на экране; отчёт сохраняет печатную палитру."""
    background, foreground, muted = "#181818", "#e8ece5", "#a6b09f"
    palette = {
        "#1f77b4": "#b6ef95", "#0000ff": "#b6ef95",
        "#ff7f0e": "#e5c58a", "#2ca02c": "#8dc6bd",
        "#d62728": "#ee9a8b", "#ff0000": "#ee9a8b",
        "#9467bd": "#b9b3e2",
    }

    def terminal_color(color):
        rgba = to_rgba(color)
        return palette.get(to_hex(rgba), color) if rgba[3] else color

    figure.set_facecolor(background)
    for artist in figure.findobj():
        if isinstance(artist, Line2D):
            artist.set_color(terminal_color(artist.get_color()))
        elif isinstance(artist, Patch):
            artist.set_facecolor(terminal_color(artist.get_facecolor()))
        elif isinstance(artist, Text):
            artist.set_color(foreground)
            artist.set_fontfamily("DejaVu Sans Mono")
    for axis in figure.axes:
        axis.set_facecolor(background)
        axis.set_title(fill(axis.get_title(), 56), fontsize=10, color=foreground, pad=12)
        axis.tick_params(colors=muted, labelsize=9)
        axis.xaxis.label.set_color(muted)
        axis.yaxis.label.set_color(muted)
        for spine in axis.spines.values():
            spine.set_color("#424a40")
        for line in axis.get_xgridlines() + axis.get_ygridlines():
            line.set_color("#424a40")
            line.set_alpha(.6)
        legend = axis.get_legend()
        if legend:
            legend.get_frame().set_facecolor(background)
            legend.get_frame().set_edgecolor("#424a40")
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True, width="stretch", alt=title or "График энергетической модели")
    plt.close(figure)


def show_p2p_chart(data: pd.DataFrame, *, title: str, xlabel: str, ylabel: str, **style):
    figure = make_analysis_figure(data, title=title, xlabel=xlabel, ylabel=ylabel, **style)
    show_terminal_plot(figure, title=title)


configs = get_configs(tuple(path.stat().st_mtime for path in CONFIG_PATHS.values()))

# =====================================================================
# БОКОВАЯ ПАНЕЛЬ УПРАВЛЕНИЯ ПАРАМЕТРАМИ
# =====================================================================
st.sidebar.title("[ Параметры узла ]")
st.sidebar.caption("INPUT / Измените параметры для пересчёта")

calibrated_is_placeholder = configs["calibrated"].get("data_status") == "placeholder"
calibrated_label = (
    "Откалиброванная (шаблон, до измерений)" if calibrated_is_placeholder else "Откалиброванная (стенд DXL-IoT)"
)

model_choice = st.sidebar.radio(
    "Тип модели (Гипотеза Г6):",
    options=["calibrated", "apriori"],
    format_func=lambda x: calibrated_label if x == "calibrated" else "Априорная (по даташитам)",
)
active_cfg = configs[model_choice]
if model_choice == "calibrated" and calibrated_is_placeholder:
    st.sidebar.warning(
        "calibrated: шаблон до измерений С0–С8. Значения условные, это не результаты измерений."
    )

mcu_choice = st.sidebar.selectbox(
    "Аппаратная конфигурация:",
    options=["K1_ESP32", "K2_ATMEGA"],
    format_func=lambda x: "К1: ESP32 (основной)" if x == "K1_ESP32" else "К2: ATmega328P (+ ESP32 в гибернации)",
)

if mcu_choice == "K1_ESP32":
    cpu_freq = st.sidebar.select_slider("Частота ЦП ESP32, МГц:", options=[80, 160, 240], value=80)
    sleep_mode = st.sidebar.selectbox("Режим сна МК:", options=["deep_sleep", "light_sleep", "hibernation"], index=0)
else:
    cpu_freq = st.sidebar.select_slider("Частота ЦП ATmega, МГц:", options=[2, 4, 8], value=4)
    sleep_mode = st.sidebar.selectbox("Режим сна МК:", options=["power_down", "power_save", "idle"], index=0)

radio_options = list(active_cfg["radio_modules"].keys())
default_radio = active_cfg["experiment_grid"]["baseline"].get("radio_module")
default_radio_index = radio_options.index(default_radio) if default_radio in radio_options else 0
radio_module = st.sidebar.selectbox(
    "Радиомодуль:",
    options=radio_options,
    index=default_radio_index,
)
supported_band_plans = get_supported_band_plans(active_cfg, radio_module)
radio_cfg = active_cfg["radio_modules"][radio_module]
st.sidebar.caption(
    f"{radio_cfg.get('model_name') or radio_module} · {radio_cfg.get('chipset') or 'чип не указан'} · "
    f"питание {radio_cfg['supply_voltage_v']:g} В"
)
route = ("SENSOR --> E220 / P2P --> BRIDGE --> LoRaWAN"
         if radio_module == "EBYTE" else "BME280 --> MCU --> LoRaWAN --> GATEWAY")
model_status = f"{model_choice} / шаблон" if model_choice == "calibrated" and calibrated_is_placeholder else model_choice
st.html(f"""
<section class="terminal-header" aria-label="Модель энергопотребления LoRaWAN">
  <div class="terminal-kicker">DXL-IoT / АНАЛИТИЧЕСКАЯ МОДЕЛЬ</div>
  <h1><span class="terminal-prompt">&gt;</span> LoRaWAN / модель<span class="terminal-cursor">_</span></h1>
  <p class="terminal-description">Энергопотребление, радиоканал и автономность сенсорного узла.</p>
  <div class="terminal-route">{escape(route)}</div>
  <div class="terminal-status">
    <span>[MODEL] <b>{escape(model_status)}</b></span>
    <span>[RADIO] <b>{escape(radio_cfg.get('model_name') or radio_module)}</b></span>
    <span>[MCU] <b>{escape(mcu_choice)} / {cpu_freq} MHz</b></span>
  </div>
  <div class="terminal-footer" aria-hidden="true"></div>
</section>
""")
if radio_module == "EBYTE" and "EBYTE" in active_cfg.get("p2p_links", {}):
    st.sidebar.subheader("Канал E220 P2P")
    profile_key = st.sidebar.selectbox(
        "Параметры E220:", options=list(active_cfg["p2p_links"]["EBYTE"]["parameter_profiles"]),
        index=list(active_cfg["p2p_links"]["EBYTE"]["parameter_profiles"]).index(active_cfg["p2p_links"]["EBYTE"]["parameter_profile"]),
        format_func=lambda key: {"legacy": "Старая версия: RX 16,8 мА", "v1_7": "Руководство v1.7: RX ≈8 мА"}.get(key, key),
    )
    p2p_profile = active_cfg["p2p_links"]["EBYTE"]["parameter_profiles"][profile_key]
    p2p_channel = st.sidebar.number_input("Канал E220 (0–83):", min_value=0, max_value=83, value=23, step=1)
    p2p_power = st.sidebar.selectbox("Мощность E220, дБм:", options=[22, 17, 13, 10])
    p2p_rate = st.sidebar.selectbox("Скорость E220, бит/с:", options=active_cfg["p2p_links"]["EBYTE"]["air_rate_levels_bps"])
    p2p_data_limit = min(51, active_cfg["p2p_links"]["EBYTE"]["lorawan_forward_payload_limit_bytes"],
                         active_cfg["p2p_links"]["EBYTE"]["subpacket_bytes"]) - ENVELOPE_BYTES
    p2p_bytes = st.sidebar.number_input("Данные сенсора, байт:", min_value=1, max_value=p2p_data_limit, value=min(12, p2p_data_limit), step=1)
    p2p_period = st.sidebar.number_input("Период отправки P2P, с:", min_value=1.0, max_value=86400.0, value=600.0, step=1.0)
    p2p_distance = st.sidebar.number_input("Сенсор → мост, км:", min_value=0.01, max_value=100.0, value=0.5, step=0.1)
    p2p_sensitivity = st.sidebar.number_input(
        "Чувствительность E220 на мосту, дБм:", min_value=-180.0, max_value=-50.0,
        value=float(p2p_profile["sensitivity_dbm"]), step=0.5, key=f"p2p_sensitivity_{profile_key}",
    )
    measured_tx = st.sidebar.number_input(
        "Измеренное время TX E220, мс (0 — оценка):", min_value=0.0, value=0.0, step=1.0,
        help="Время именно радио-TX, без UART и пробуждения; оценка по скорости не заменяет осциллограмму.",
    )
    radio_overhead = st.sidebar.number_input(
        "Оценка служебных радиобайтов E220:", min_value=0, max_value=255, value=0, step=1,
        help="Неизвестный overhead прошивки. Ноль означает, что он не учтён; контейнер ID/sequence/CRC32 добавляется отдельно.",
    )
    p2p_battery = st.sidebar.selectbox("Батарея сенсора и моста:", options=list(active_cfg["batteries"]))
    st.sidebar.subheader("Мост → LoRaWAN")
    bridge_modules = [key for key in active_cfg["radio_modules"] if get_supported_band_plans(active_cfg, key)]
    bridge_default = bridge_modules.index("DRF1278F") if "DRF1278F" in bridge_modules else 0
    bridge_radio = st.sidebar.selectbox("Радиомодуль LoRaWAN на мосту:", options=bridge_modules, index=bridge_default)
    bridge_plan = st.sidebar.selectbox("План LoRaWAN моста:", options=get_supported_band_plans(active_cfg, bridge_radio))
    bridge_mcu = st.sidebar.selectbox("МК моста:", options=list(active_cfg["mcu_configs"]),
                                    key="p2p_bridge_mcu")
    bridge_mcu_cfg = active_cfg["mcu_configs"][bridge_mcu]
    bridge_cpu = st.sidebar.selectbox("Частота МК моста, МГц:",
                                     options=sorted(bridge_mcu_cfg["active_current_by_freq_ma"]),
                                     index=sorted(bridge_mcu_cfg["active_current_by_freq_ma"]).index(bridge_mcu_cfg["base_freq_mhz"]),
                                     key=f"p2p_bridge_cpu_{bridge_mcu}")
    bridge_sleep = st.sidebar.selectbox("Сон МК моста:", options=list(bridge_mcu_cfg["sleep_modes"]),
                                       index=list(bridge_mcu_cfg["sleep_modes"]).index("deep_sleep" if bridge_mcu == "K1_ESP32" else "power_down"),
                                       key=f"p2p_bridge_sleep_{bridge_mcu}")
    bridge_wait = st.sidebar.radio("Ожидание RX моста:", options=["light_sleep", "active"],
                                  key="p2p_bridge_wait")
    bridge_class = st.sidebar.radio("Класс LoRaWAN моста:", options=["A", "B"], horizontal=True,
                                   key="p2p_bridge_class")
    bridge_ping = st.sidebar.select_slider("Ping-слоты моста k:", options=[7, 4, 1], value=4) if bridge_class == "B" else None
    bridge_adr = st.sidebar.checkbox("ADR на участке мост → шлюз", value=False)
    bridge_sf = st.sidebar.select_slider("SF моста:", options=[7, 8, 9, 10, 11, 12], value=9)
    bridge_distance = st.sidebar.number_input("Мост → LoRaWAN-шлюз, км:", min_value=0.01, max_value=100.0, value=0.5, step=0.1)
    bridge_radio_cfg = active_cfg["radio_modules"][bridge_radio]
    bridge_plan_cfg = resolve_band_plan(active_cfg, bridge_plan)[1]
    bridge_channel_cfg = bridge_plan_cfg["channel_adr"]
    bridge_system = bridge_plan_cfg["system"]
    st.sidebar.caption(
        f"LoRaWAN: {bridge_system['frequency_hz'] / 1e6:.3f} МГц; "
        f"RX2 {bridge_system['rx2_frequency_hz'] / 1e6:.3f} МГц / SF{bridge_system['rx2_sf']}"
    )
    bridge_rx_sensitivity = st.sidebar.number_input(
        "Чувствительность LoRaWAN-радио моста, дБм:", min_value=-180.0, max_value=-50.0,
        value=float(bridge_radio_cfg["sensitivity_dbm"]), step=.5,
        help=(f"Опорный SF{bridge_radio_cfg['sensitivity_reference_sf']}/"
              f"BW{bridge_radio_cfg['sensitivity_reference_bandwidth_khz']}. "
              + (bridge_radio_cfg.get("sensitivity_conditions") or "")),
        key=f"p2p_bridge_rx_{model_choice}_{bridge_radio}",
    )
    bridge_gateway_default = bridge_channel_cfg.get("gateway_sensitivity_dbm")
    if bridge_gateway_default is None:
        bridge_gateway_default = calc_noise_floor_dbm(
            bw_khz=float(bridge_channel_cfg.get("gateway_sensitivity_reference_bandwidth_khz", 125)),
            noise_figure_db=float(bridge_channel_cfg["gateway_noise_figure_db"]),
        ) + bridge_channel_cfg["snr_thresholds_db"][12]
    bridge_gateway_sensitivity = st.sidebar.number_input(
        "Чувствительность LoRaWAN-шлюза, дБм:", min_value=-180.0, max_value=-50.0,
        value=float(bridge_gateway_default), step=.5, key=f"p2p_gateway_rx_{model_choice}_{bridge_plan}",
        help="Порог шлюза в опорном SF/BW частотного плана; пересчитывается для выбранного SF.",
    )
    st.caption("[ROUTE] Сенсор E220 --> LoRa P2P --> мост --> LoRaWAN")
    st.info(
        "Сенсор передаёт данные второму E220. Мост с отдельным МК и радиомодулем "
        "отправляет их как полезную нагрузку своего LoRaWAN-устройства. "
        "Это прикладной мост, не прямое подключение E220 к LoRaWAN-шлюзу."
    )
    p2p_parameters = dict(
        channel=p2p_channel, tx_power_dbm=p2p_power, air_rate_bps=p2p_rate,
        payload_bytes=p2p_bytes, tx_period_s=p2p_period, distance_km=p2p_distance,
        measured_airtime_ms=measured_tx if measured_tx > 0 else None,
        assumed_radio_overhead_bytes=radio_overhead, parameter_profile=profile_key,
        sensitivity_dbm=p2p_sensitivity, mcu_config=mcu_choice, cpu_freq_mhz=float(cpu_freq),
        sleep_mode=sleep_mode, battery_type=p2p_battery, bridge_radio_module=bridge_radio,
        bridge_band_plan=bridge_plan, bridge_sf=bridge_sf, bridge_distance_km=bridge_distance,
        bridge_mcu_config=bridge_mcu, bridge_cpu_freq_mhz=float(bridge_cpu),
        bridge_sleep_mode=bridge_sleep, bridge_wait_mode=bridge_wait,
        bridge_device_class=bridge_class, bridge_ping_periodicity_k=bridge_ping,
        bridge_adr_enabled=bridge_adr, bridge_receiver_sensitivity_dbm=bridge_rx_sensitivity,
        bridge_gateway_sensitivity_dbm=bridge_gateway_sensitivity,
    )
    try:
        p2p_result = evaluate_p2p_bridge(active_cfg, **p2p_parameters)
    except ValueError as err:
        st.error(str(err))
        st.stop()
    st.caption(
        f"E220: {p2p_result.frequency_hz / 1e6:.3f} МГц / {p2p_rate} бит/с / {p2p_power} дБм. "
        f"Мост: {bridge_radio} / {bridge_plan} / SF{p2p_result.bridge_lorawan.sf}. "
        f"Контейнер: {p2p_result.forwarded_bytes} байт, включая {ENVELOPE_BYTES} служебных."
    )
    c_sensor, c_bridge = st.columns(2)
    with c_sensor:
        st.subheader("Автономный сенсор")
        st.metric("Средний ток сенсора", f"{p2p_result.sensor_avg_current_ma:.3f} мА")
        st.metric("Срок работы сенсора", f"{p2p_result.sensor_battery.lifetime_days:.1f} сут.")
        st.metric("Энергия периода сенсора", f"{p2p_result.sensor_cycle_energy_mj:.2f} мДж")
    with c_bridge:
        st.subheader("Мост: E220 RX + МК + LoRaWAN")
        st.metric("Средний ток моста", f"{p2p_result.bridge_avg_current_ma:.3f} мА")
        st.metric("Срок работы моста от такой же батареи", f"{p2p_result.bridge_battery.lifetime_days:.1f} сут.")
        st.metric("Энергия периода моста", f"{p2p_result.bridge_cycle_energy_mj:.2f} мДж")
    st.caption(
        f"Токи приведены к шине {p2p_result.reference_voltage_v:g} В; радио E220 питается от "
        f"{radio_cfg['supply_voltage_v']:g} В. МК моста — {bridge_mcu}, класс {bridge_class}. "
        "Один опрос BME280 на пакет у сенсора. Энергия LoRaWAN-шлюза и сервера не включена."
    )
    p2p_tables = p2p_result_tables(p2p_result)
    phase_df = p2p_tables["sensor_phases"].rename(columns={
        "state": "Состояние", "duration_ms": "Время, мс", "radio_mj": "Радио, мДж",
        "mcu_sensor_mj": "МК и датчик, мДж", "total_mj": "Всего, мДж",
        "equivalent_current_ma": "Эквивалентный ток, мА",
    })
    st.write(
        f"Время TX E220: {p2p_result.airtime_ms:.1f} мс "
        f"({'измеренное' if p2p_result.airtime_source == 'measured' else 'оценочное'}). "
        f"Запас P2P: {p2p_result.p2p_margin_db:.1f} дБ; "
        f"запас uplink LoRaWAN: {p2p_result.bridge_lorawan.adr_info.link_margin_db:.1f} дБ."
    )
    if not p2p_result.end_to_end_link_feasible:
        st.warning("Бюджет линии не обеспечивает хотя бы один из участков маршрута.")
    p2p_tabs = st.tabs([
        "[01] Профили / Г1", "[02] Сон и период / Г2, Г6",
        "[03] МК и класс / Г3, Г5", "[04] Дальность / Г4", "[05] Чувствительность / ПФЭ",
    ], key="p2p_analysis_tab", on_change="rerun")
    if p2p_tabs[0].open:
        with p2p_tabs[0]:
            bridge_energy = p2p_tables["bridge_energy"].rename(columns={"component": "Составляющая", "energy_mj": "Энергия, мДж"})
            c_profile, c_budget = st.columns(2)
            with c_profile:
                st.subheader("Профиль сенсора P2P")
                show_p2p_chart(phase_df.set_index("Состояние")[["Радио, мДж", "МК и датчик, мДж"]],
                               title="Энергия сенсора по состояниям", xlabel="Состояние", ylabel="Энергия, мДж",
                               kind="bar", stacked=True)
                st.dataframe(phase_df, hide_index=True)
            with c_budget:
                st.subheader("Полный бюджет моста")
                show_p2p_chart(bridge_energy.set_index("Составляющая"), title="Энергетический бюджет моста",
                               xlabel="Составляющая", ylabel="Энергия, мДж", kind="bar")
                st.dataframe(bridge_energy, hide_index=True)
            with st.expander("Состояния LoRaWAN-радио и МК моста"):
                st.dataframe(p2p_tables["bridge_states"], hide_index=True)
                st.caption("Токи физических шин. Маяки и ping-слоты — повторяющиеся события; полный бюджет учитывает их частоту.")
            cycle = get_p2p_analysis(active_cfg, p2p_parameters, "cycle")
            st.subheader("Г1: влияние SF на отдельное LoRaWAN-радио моста")
            sf_table = cycle["bridge_sf"]
            left, right = st.columns(2)
            with left:
                show_p2p_chart(sf_table.set_index("sf")[["bridge_toa_ms", "bridge_transaction_mj"]],
                               title="Время TX и энергия транзакции LoRaWAN", xlabel="Коэффициент SF моста",
                               ylabel="Время TX, мс", colors=("#1f77b4", "#d62728"),
                               secondary_y="bridge_transaction_mj", secondary_ylabel="Энергия транзакции, мДж")
                st.caption("SF меняется только у моста; E220 сохраняет выбранную скорость P2P.")
            with right:
                if "p2p_rate" in cycle:
                    show_p2p_chart(cycle["p2p_rate"].set_index("rate_bps")[["sensor_avg_ma"]],
                                   title="Влияние скорости E220", xlabel="Скорость E220, бит/с",
                                   ylabel="Средний ток сенсора, мА")
                    st.dataframe(cycle["p2p_rate"][["rate_bps", "airtime_ms", "sensor_avg_ma", "sensor_lifetime_days", "error"]], hide_index=True)
                    st.caption("Время по скорости — приближение. Чувствительность при других скоростях требует измерений.")
                else:
                    st.info("Перебор скорости отключён: измеренное время TX относится только к выбранному пакету и скорости.")
    if p2p_tabs[1].open:
        with p2p_tabs[1]:
            st.subheader("Г2: сон сенсора и период передачи")
            sleep_table = get_p2p_analysis(active_cfg, p2p_parameters, "sleep")["sleep_period"]
            deep_key = "deep_sleep" if mcu_choice == "K1_ESP32" else "power_down"
            shallow_key = "light_sleep" if mcu_choice == "K1_ESP32" else "idle"
            c_sleep_sensor, c_sleep_bridge = st.columns(2)
            with c_sleep_sensor:
                show_p2p_chart(sleep_table.pivot(index="period_min", columns="sleep_mode", values="sensor_lifetime_days")[[deep_key, shallow_key]],
                               title="Влияние сна и периода: сенсор", xlabel="Период передачи, мин",
                               ylabel="Срок работы, сут.", log_x=True)
                st.caption("Один опрос датчика на каждую отправку.")
            with c_sleep_bridge:
                show_p2p_chart(sleep_table.pivot(index="period_min", columns="sleep_mode", values="bridge_lifetime_days")[[deep_key, shallow_key]],
                               title="Влияние периода: мост", xlabel="Период передачи, мин",
                               ylabel="Срок работы, сут.", log_x=True)
                st.caption("E220 моста непрерывно слушает независимо от периода; режим сна меняется у сенсора.")
            sensor_modes = active_cfg["mcu_configs"][mcu_choice]["sleep_modes"]
            t_be = calc_sleep_breakeven_time_s(
                sensor_modes[shallow_key]["wakeup_energy_uj"], sensor_modes[deep_key]["wakeup_energy_uj"],
                sensor_modes[shallow_key]["current_ma"] * 1000, sensor_modes[deep_key]["current_ma"] * 1000,
                voltage_v=p2p_result.reference_voltage_v,
            )
            st.caption(f"Порог окупаемости пробуждения МК: {t_be:.3f} с ({shallow_key} → {deep_key}).")
            st.subheader("Г6: сравнение априорной и откалиброванной моделей")
            comparison_rows = []
            for name, cfg in configs.items():
                try:
                    r = evaluate_p2p_bridge(cfg, **p2p_parameters)
                    comparison_rows.append({"Модель": name, "Ток сенсора, мА": r.sensor_avg_current_ma,
                                            "Ток моста, мА": r.bridge_avg_current_ma,
                                            "Сенсор, сут.": r.sensor_battery.lifetime_days,
                                            "Мост, сут.": r.bridge_battery.lifetime_days})
                except ValueError as err:
                    st.warning(f"{name}: {err}")
            st.dataframe(pd.DataFrame(comparison_rows), hide_index=True)
            if calibrated_is_placeholder:
                st.warning("Откалиброванная модель содержит условные значения. Это сравнение допущений до измерений.")
    if p2p_tabs[2].open:
        with p2p_tabs[2]:
            hardware = get_p2p_analysis(active_cfg, p2p_parameters, "hardware")
            c_cpu_sensor, c_cpu_bridge = st.columns(2)
            with c_cpu_sensor:
                st.subheader("Частота МК сенсора")
                show_p2p_chart(hardware["sensor_cpu"].set_index("cpu_mhz")[["sensor_avg_ma"]],
                               title="Влияние частоты МК сенсора", xlabel="Частота МК, МГц", ylabel="Средний ток сенсора, мА")
                st.caption("В UART и TX E220 МК сенсора активен; частота изменяет ток и длительность подготовки.")
            with c_cpu_bridge:
                st.subheader("Г5: ожидание RX1/RX2 моста")
                show_p2p_chart(hardware["bridge_wait_cpu"].pivot(index="cpu_mhz", columns="wait_mode", values="bridge_avg_ma")[["light_sleep", "active"]],
                               title="Режим ожидания RX и частота МК моста", xlabel="Частота МК, МГц",
                               ylabel="Средний ток моста, мА", kind="bar")
            st.subheader("Г3: класс A/B отдельного LoRaWAN-радио моста")
            c_class, _ = st.columns(2)
            with c_class:
                class_chart = hardware["bridge_class"].set_index("class")
                class_chart = pd.DataFrame({
                    "Базовый расход класса A": class_chart["bridge_avg_ma"] - class_chart["bridge_class_addon_ma"],
                    "Надбавка класса B": class_chart["bridge_class_addon_ma"],
                })
                show_p2p_chart(class_chart,
                               title="Влияние класса LoRaWAN на мост", xlabel="Класс и периодичность ping-слотов",
                               ylabel="Средний ток моста, мА", kind="bar", stacked=True, colors=("#2ca02c", "#d62728"))
            st.dataframe(hardware["bridge_class"][["class", "bridge_avg_ma", "bridge_lifetime_days", "end_to_end_link_feasible", "error"]], hide_index=True)
            st.caption("Класс B добавляет маяки и ping-слоты моста; бюджет сенсора E220 от этого не меняется.")
    if p2p_tabs[3].open:
        with p2p_tabs[3]:
            ranges = get_p2p_analysis(active_cfg, p2p_parameters, "range")
            st.subheader("Запас линии текущего маршрута")
            st.dataframe(p2p_tables["link_margins"].rename(columns={"link": "Участок / окно", "margin_db": "Запас, дБ"}), hide_index=True)
            st.subheader("Сенсор → мост: дальность P2P")
            margin_table = ranges["p2p_distance"][["distance_km", "p2p_margin_db"]].copy()
            margin_table["Порог связи, дБ"] = 0.0
            c_p2p_range, _ = st.columns(2)
            with c_p2p_range:
                show_p2p_chart(margin_table.set_index("distance_km"), title="Запас линии сенсор → мост",
                               xlabel="Расстояние до моста, км", ylabel="Запас линии, дБ", colors=("#1f77b4", "#d62728"))
            st.caption("Порог E220 задан в панели. Фиксированные мощность и скорость не меняют энергию с расстоянием; потери пакетов и повторы не моделируются.")
            st.subheader("Г4: мост → шлюз, ADR против фиксированного SF12")
            adr_table = ranges["bridge_adr"]
            feasible_adr = adr_table.copy()
            feasible_adr.loc[~feasible_adr["end_to_end_link_feasible"].fillna(False).astype(bool), "bridge_avg_ma"] = np.nan
            c_adr_current, c_adr_sf = st.columns(2)
            with c_adr_current:
                show_p2p_chart(feasible_adr.pivot(index="distance_km", columns="mode", values="bridge_avg_ma")[["ADR", "SF12"]],
                               title="Влияние ADR на потребление моста", xlabel="Расстояние до шлюза, км",
                               ylabel="Средний ток моста, мА", colors=("#1f77b4", "#d62728"))
            with c_adr_sf:
                sf_adr = adr_table[adr_table["mode"] == "ADR"].set_index("distance_km")[["bridge_sf"]].rename(columns={"bridge_sf": "ADR"})
                show_p2p_chart(sf_adr, title="Выбор SF алгоритмом ADR", xlabel="Расстояние до шлюза, км",
                               ylabel="Установившийся SF", kind="step", colors=("#9467bd",))
            st.dataframe(adr_table[["distance_km", "mode", "bridge_sf", "bridge_tx_power_dbm", "bridge_avg_ma", "end_to_end_link_feasible", "error"]], hide_index=True)
            st.caption("На графике тока показаны точки, обеспечивающие весь маршрут; таблица сохраняет недоступные точки с отметкой False. ADR действует только на LoRaWAN-радио моста.")
    if p2p_tabs[4].open:
        with p2p_tabs[4]:
            factors = get_p2p_analysis(active_cfg, p2p_parameters, "factors")
            st.subheader("Влияние факторов на срок работы")
            sensitivity_table = factors["sensitivity"]
            st.dataframe(sensitivity_table[["factor", "base", "varied", "sensor_lifetime_change_pct", "bridge_lifetime_change_pct", "error"]], hide_index=True)
            c_sens_sensor, c_sens_bridge = st.columns(2)
            with c_sens_sensor:
                show_p2p_chart(sensitivity_table.set_index("factor")[["sensor_lifetime_change_pct"]],
                               title="Чувствительность: сенсор", xlabel="Изменение срока работы, %", ylabel="Фактор", kind="barh")
            with c_sens_bridge:
                show_p2p_chart(sensitivity_table.set_index("factor")[["bridge_lifetime_change_pct"]],
                               title="Чувствительность: мост", xlabel="Изменение срока работы, %", ylabel="Фактор", kind="barh")
            st.caption("Каждый фактор изменяется отдельно от текущей точки. Проценты показывают изменение срока работы двух независимых батарей.")
            st.subheader("ПФЭ 2³: период, частота МК и сон сенсора")
            st.write("x₁: T → 2T; x₂: минимальная → максимальная частота МК сенсора; x₃: глубокий → неглубокий сон. Отклики: lg(I_avg, мА) сенсора и моста.")
            st.dataframe(factors["pfe_design"][["x1", "x2", "x3", "period_s", "cpu_mhz", "sleep_mode", "sensor_avg_ma", "bridge_avg_ma", "error"]], hide_index=True)
            if factors["pfe_coefficients"].empty:
                st.warning("Часть точек не помещается в период; увеличьте период для расчёта коэффициентов ПФЭ.")
            else:
                st.dataframe(factors["pfe_coefficients"], hide_index=True)
                c_coefficients, _ = st.columns(2)
                with c_coefficients:
                    show_p2p_chart(factors["pfe_coefficients"].set_index("term").drop(index="b0"),
                                   title="Коэффициенты ПФЭ 2³", xlabel="Член регрессии", ylabel="Коэффициент для lg(I_avg)", kind="bar")
            st.caption("Это детерминированный вычислительный эксперимент. Критерии Кохрена и Стьюдента требуют повторных измерений и здесь не рассчитываются.")
    with st.expander("Контейнер данных для прикладного моста"):
        sensor_id = st.number_input("ID сенсора:", min_value=0, max_value=65535, value=1, step=1)
        sequence = st.number_input("Номер сообщения:", min_value=0, max_value=4294967295, value=0, step=1)
        sample_frame = encode_p2p_message(sensor_id, sequence, bytes(p2p_bytes))
        st.code(sample_frame.hex(" "), language="text")
        st.caption(
            "Пример с нулевыми данными: версия, ID, sequence, длина, данные, CRC32. "
            f"Мост передаёт контейнер в FRMPayload на FPort {p2p_result.forward_fport}. "
            "Сессия, DevEUI и ключи LoRaWAN принадлежат мосту."
        )
    st.download_button("Скачать расчёт маршрута CSV", p2p_tables["summary"].to_csv(index=False).encode("utf-8-sig"),
                       file_name="p2p_lorawan_bridge.csv", mime="text/csv")
    st.download_button("Скачать профиль сенсора CSV", phase_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="p2p_sensor_phases.csv", mime="text/csv")
    if st.button("Подготовить полный отчёт P2P", help="Все разделы анализа, исходные параметры, CSV и графики PNG/PDF."):
        with st.spinner("Готовлю таблицы и графики отчёта…"):
            report_bytes = get_p2p_report(active_cfg, p2p_parameters, configs)
        st.download_button("Скачать отчёт P2P ZIP", report_bytes,
                           file_name="p2p_report_bundle.zip", mime="application/zip", on_click="ignore")
    with st.expander("Границы расчёта и допущения"):
        for warning in p2p_result.warnings:
            st.write(warning)
        st.write("Включение модуля в модель не заменяет разработку и проверку прошивок сенсора и моста.")
    st.stop()
if not supported_band_plans:
    st.warning(get_radio_compatibility_note(radio_cfg))
    st.stop()
band_plan_details = {
    name: resolve_band_plan(active_cfg, name)[1]
    for name in supported_band_plans
}
default_band_plan = active_cfg["experiment_grid"]["baseline"].get("band_plan")
if default_band_plan not in supported_band_plans:
    default_band_plan = supported_band_plans[0]
band_plan = st.sidebar.selectbox(
    "Частотный план LoRaWAN:",
    options=supported_band_plans,
    index=supported_band_plans.index(default_band_plan),
    format_func=lambda name: band_plan_details[name].get("label", name),
)
selected_system_cfg = band_plan_details[band_plan]["system"]
selected_channel_cfg = band_plan_details[band_plan]["channel_adr"]
receiver_sensitivity = st.sidebar.number_input(
    f"Чувствительность узла при SF{radio_cfg['sensitivity_reference_sf']}/BW{radio_cfg['sensitivity_reference_bandwidth_khz']}, дБм:",
    min_value=-180.0, max_value=-50.0, value=float(radio_cfg['sensitivity_dbm']), step=0.5,
    help=radio_cfg.get('sensitivity_conditions'), key=f"node_sensitivity_{model_choice}_{radio_module}",
)
gateway_default_sensitivity = selected_channel_cfg.get("gateway_sensitivity_dbm")
if gateway_default_sensitivity is None:
    gateway_default_sensitivity = calc_noise_floor_dbm(
        bw_khz=float(selected_channel_cfg.get("gateway_sensitivity_reference_bandwidth_khz", 125)),
        noise_figure_db=float(selected_channel_cfg['gateway_noise_figure_db']),
    ) + selected_channel_cfg['snr_thresholds_db'][12]
gateway_sensitivity = st.sidebar.number_input(
    "Чувствительность шлюза при SF12/BW125, дБм:",
    min_value=-180.0, max_value=-50.0, value=float(gateway_default_sensitivity), step=0.5,
    help="Порог приёма uplink. Чувствительность узла используется отдельно для RX1/RX2.",
    key=f"gateway_sensitivity_{model_choice}_{band_plan}",
)
radio_overrides = dict(
    radio_module=radio_module, band_plan=band_plan,
    receiver_sensitivity_dbm=receiver_sensitivity, gateway_sensitivity_dbm=gateway_sensitivity,
)
st.sidebar.caption(
    " / ".join(
        [
            f"Регион: {selected_system_cfg['region']}",
            f"Частота: {selected_system_cfg['frequency_hz'] / 1_000_000:.3f} МГц",
            f"RX2: {selected_system_cfg['rx2_frequency_hz'] / 1e6:.3f} МГц / SF{selected_system_cfg['rx2_sf']}",
        ]
    )
)
wait_mode = st.sidebar.radio(
    "Ожидание окон RX1/RX2 (Г5):",
    options=["light_sleep", "active"],
    format_func=lambda x: "Неглубокий сон (Light/Idle)" if x == "light_sleep" else "Активное ожидание (Active)",
)

st.sidebar.markdown("---")
st.sidebar.subheader("[ Радиоинтерфейс и трафик ]")

adr_enabled = st.sidebar.checkbox("Включить механизм ADR (Г4)", value=False)
distance_km = st.sidebar.slider("Расстояние до шлюза, км:", min_value=0.05, max_value=15.0, value=0.5, step=0.05)
sf_input = st.sidebar.select_slider("Коэффициент SF (при выкл. ADR):", options=[7, 8, 9, 10, 11, 12], value=9)
payload_bytes = st.sidebar.select_slider("Полезная нагрузка L, байт:", options=[4, 12, 24, 51], value=12)
tx_period_min = st.sidebar.slider("Период передачи T_tx, мин:", min_value=1, max_value=720, value=10)

sensor_freq_label = st.sidebar.selectbox(
    "Частота опроса BME280:",
    options=["1/60 Гц (раз в мин)", "0.1 Гц (раз в 10 с)", "1 Гц (каждую с)"],
)
sensor_freq_map = {
    "1/60 Гц (раз в мин)": 1.0 / 60.0,
    "0.1 Гц (раз в 10 с)": 0.1,
    "1 Гц (каждую с)": 1.0,
}
sensor_freq_hz = sensor_freq_map[sensor_freq_label]

device_class = st.sidebar.radio("Класс устройства LoRaWAN (Г3):", options=["A", "B"], horizontal=True)
ping_k = None
if device_class == "B":
    ping_k = st.sidebar.select_slider("Периодичность ping-слотов k:", options=[7, 4, 1], value=4)

battery_type = st.sidebar.selectbox(
    "Источник питания:",
    options=["li_socl2_aa", "li_ion_18650", "ideal"],
    format_func=lambda x: {
        "li_socl2_aa": "Li-SOCl2 AA (2400 мА·ч, 1%/год)",
        "li_ion_18650": "Li-ion 18650 (2600 мА·ч, 3%/мес)",
        "ideal": "Идеальный (2400 мА·ч, без саморазряда)",
    }[x],
)

# Расчёт текущей точки
res = evaluate_node_energy_budget(
    active_cfg,
    sf=sf_input,
    adr_enabled=adr_enabled,
    distance_km=distance_km,
    payload_bytes=payload_bytes,
    sensor_freq_hz=sensor_freq_hz,
    mcu_config=mcu_choice,
    cpu_freq_mhz=float(cpu_freq),
    sleep_mode=sleep_mode,
    wait_mode=wait_mode,
    device_class=device_class,
    ping_periodicity_k=ping_k,
    battery_type=battery_type,
    tx_period_s=float(tx_period_min) * 60.0,
    **radio_overrides,
)

# =====================================================================
# ГЛАВНЫЙ ЭКРАН: СВОДНЫЕ КАРТОЧКИ МЕТРИК
# =====================================================================
st.caption("[RESULT] Энергетический бюджет LoRaWAN-узла")
radio_information_notes = {get_radio_compatibility_note(radio_cfg), radio_cfg.get("tx_current_note")}
for model_warning in res.model_warnings:
    if model_warning not in radio_information_notes:
        st.warning(model_warning)
st.caption(
    f"Питание радиомодуля: {res.radio_supply_voltage_v:g} В. "
    f"Суммарные токи и срок службы приведены к {res.profile.voltage_v:g} В; "
    "КПД преобразователя задан в параметрах радиомодуля (по умолчанию идеальный)."
)
st.caption(
    f"Конфигурация: **{mcu_choice}** ({cpu_freq} МГц) | Радиомодуль: **{radio_module}** | "
    f"Band plan: **{res.band_plan}** | "
    f"Фактический **SF{res.sf}** ($P_{{tx}} = {res.tx_power_dbm:.0f}$ дБм) | "
    f"Класс **{res.device_class}**"
)

with st.container(key="terminal_metrics"):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Срок службы", f"{res.battery.lifetime_years:.2f} лет", f"{res.battery.lifetime_days:.0f} сут.", delta_color="off", delta_arrow="off")
    c2.metric("Средний ток I_avg", f"{res.i_avg_total_ma * 1000.0:.1f} мкА", f"{res.i_avg_total_ma:.4f} мА", delta_color="off", delta_arrow="off")
    c3.metric("Энергия транзакции", f"{res.e_tx_rx_cycle_mj:.2f} мДж", f"t_act = {res.active_tx_rx_duration_ms:.0f} мс", delta_color="off", delta_arrow="off")
    c4.metric("Суточный бюджет E_сут", f"{res.e_day_total_j:.2f} Дж/сут", f"Сон: {res.share_sleep_pct:.1f}%", delta_color="off", delta_arrow="off")
    c5.metric("Ток сна (суммарный)", f"{res.i_sleep_total_ua:.1f} мкА", f"Ожидание RX: {res.share_wait_windows_pct:.1f}%", delta_color="off", delta_arrow="off")

st.markdown("---")

# =====================================================================
# ВКЛАДКИ ИССЛЕДОВАНИЯ И ПРОВЕРКИ ГИПОТЕЗ Г1–Г6
# =====================================================================
tabs = st.tabs([
    "[01] Профиль / Г1",
    "[02] Сон и период / Г2, Г6",
    "[03] RX и класс B / Г3, Г5",
    "[04] Дальность / Г4",
    "[05] Чувствительность / ПФЭ",
])

# ---------------------------------------------------------------------
# ВКЛАДКА 1: ПРОФИЛЬ ЦИКЛА И ГИПОТЕЗА Г1
# ---------------------------------------------------------------------
with tabs[0]:
    col_left, col_right = st.columns([1.1, 1.0])

    with col_left:
        st.subheader("Пофазовый профиль транзакции (Цепь 1 vs Цепь 2)")
        state_rows = []
        for st_enum in CLASS_A_ACTIVE_SEQUENCE:
            m_rad = res.profile.states[st_enum][PowerCircuit.RADIO]
            m_mcu = res.profile.states[st_enum][PowerCircuit.MCU_BOARD]
            state_rows.append({
                "Состояние": st_enum.value,
                "Длительность, мс": round(m_rad.duration_ms, 2),
                "I радио, мА": round(m_rad.current_ma, 3),
                "I платы МК, мА": round(m_mcu.current_ma, 3),
                "E радио, мДж": round(m_rad.energy_mj, 3),
                "E платы МК, мДж": round(m_mcu.energy_mj, 3),
                "E сумм, мДж": round(m_rad.energy_mj + m_mcu.energy_mj, 3),
            })
        df_states = pd.DataFrame(state_rows)
        st.dataframe(df_states, width="stretch", hide_index=True)

        fig1, ax1 = plt.subplots(figsize=(6.5, 3.8))
        x_idx = np.arange(len(df_states))
        ax1.bar(x_idx, df_states["E радио, мДж"], label="Цепь 1: Радиомодуль", color="#1f77b4")
        ax1.bar(
            x_idx,
            df_states["E платы МК, мДж"],
            bottom=df_states["E радио, мДж"],
            label="Цепь 2: Плата МК + датчик",
            color="#ff7f0e",
        )
        ax1.set_xticks(x_idx)
        ax1.set_xticklabels(df_states["Состояние"])
        ax1.set_ylabel("Энергия состояния, мДж")
        ax1.set_title(f"Распределение энергии транзакции по состояниям (SF{res.sf})")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)
        ax1.legend()
        show_terminal_plot(fig1)

    with col_right:
        st.subheader("Проверка гипотезы Г1: SF7..SF12 (Время в эфире vs Энергия)")
        sfs = [7, 8, 9, 10, 11, 12]
        toa_list, e_tx_rx_list, rx2_share_list = [], [], []
        for s in sfs:
            r_s = evaluate_node_energy_budget(
                active_cfg,
                sf=s,
                adr_enabled=False,
                payload_bytes=payload_bytes,
                mcu_config=mcu_choice,
                cpu_freq_mhz=float(cpu_freq),
                wait_mode=wait_mode,
                **radio_overrides,
            )
            toa_list.append(r_s.profile.states[CLASS_A_ACTIVE_SEQUENCE[1]][PowerCircuit.RADIO].duration_ms)
            e_tx_rx_list.append(r_s.e_tx_rx_cycle_mj)
            rx2_share_list.append(r_s.share_rx2_in_tx_rx_pct)

        toa_ratio = toa_list[-1] / toa_list[0]
        e_ratio = e_tx_rx_list[-1] / e_tx_rx_list[0]
        st.info(
            f"**Вывод по Г1 (при L = {payload_bytes} байт):** переход от SF7 к SF12 увеличивает "
            f"время передачи в **{toa_ratio:.1f} раза**, тогда как полная энергия радиотранзакции "
            f"возрастает лишь в **{e_ratio:.1f} раза** (за счёт фиксированных настроек RX2 выбранного плана и константных затрат на пробуждение/ожидание)."
        )

        fig2, ax2 = plt.subplots(figsize=(6.5, 4.0))
        ax2_twin = ax2.twinx()
        p1 = ax2.plot(sfs, toa_list, "o-", color="#1f77b4", linewidth=2, label="Время передачи ToA, мс")
        p2 = ax2_twin.plot(sfs, e_tx_rx_list, "s--", color="#d62728", linewidth=2, label="Энергия транзакции E_ц, мДж")
        ax2.set_xlabel("Коэффициент расширения спектра SF")
        ax2.set_ylabel("Время в эфире ToA, мс", color="#1f77b4")
        ax2_twin.set_ylabel("Энергия транзакции, мДж", color="#d62728")
        ax2.set_title("Сравнение роста времени передачи и полной энергии транзакции")
        ax2.grid(True, linestyle="--", alpha=0.5)
        lines = p1 + p2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc="upper left")
        show_terminal_plot(fig2)

# ---------------------------------------------------------------------
# ВКЛАДКА 2: РЕЖИМЫ СНА, ПЕРИОД ПЕРЕДАЧИ И КАЛИБРОВКА (Г2, Г6)
# ---------------------------------------------------------------------
with tabs[1]:
    c_g2, c_g6 = st.columns(2)

    with c_g2:
        st.subheader("Гипотеза Г2: Срок службы от периода передачи T_tx")
        periods_min = np.logspace(0, np.log10(1440), 25)
        curves = {}
        for label, kw in [
            ("SF7 + Глубокий сон", {"sf": 7, "sleep_mode": "deep_sleep" if mcu_choice == "K1_ESP32" else "power_down"}),
            ("SF12 + Глубокий сон", {"sf": 12, "sleep_mode": "deep_sleep" if mcu_choice == "K1_ESP32" else "power_down"}),
            ("SF7 + Неглубокий сон (Light/Idle)", {"sf": 7, "sleep_mode": "light_sleep" if mcu_choice == "K1_ESP32" else "idle"}),
        ]:
            curves[label] = [
                evaluate_node_energy_budget(
                    active_cfg,
                    adr_enabled=False,
                    payload_bytes=payload_bytes,
                    mcu_config=mcu_choice,
                    cpu_freq_mhz=float(cpu_freq),
                    battery_type=battery_type,
                    tx_period_s=float(p_m) * 60.0,
                    **radio_overrides,
                    **kw,
                ).battery.lifetime_years
                for p_m in periods_min
            ]

        fig_g2, ax_g2 = plt.subplots(figsize=(6.2, 4.2))
        for label, vals in curves.items():
            ax_g2.plot(periods_min, vals, linewidth=2, label=label)
        ax_g2.set_xscale("log")
        ax_g2.set_xlabel("Период передачи T_tx, мин (лог. шкала)")
        ax_g2.set_ylabel("Срок автономной работы, лет")
        ax_g2.set_title("Влияние режима сна vs SF при росте периода передачи")
        ax_g2.grid(True, which="both", linestyle="--", alpha=0.5)
        ax_g2.legend()
        show_terminal_plot(fig_g2)

        # Время безубыточности по формуле (2.1)
        esp_modes = active_cfg["mcu_configs"]["K1_ESP32"]["sleep_modes"]
        t_be = calc_sleep_breakeven_time_s(
            e_wu_shallow_uj=esp_modes["light_sleep"]["wakeup_energy_uj"],
            e_wu_deep_uj=esp_modes["deep_sleep"]["wakeup_energy_uj"],
            i_sleep_shallow_ua=esp_modes["light_sleep"]["current_ma"] * 1000.0,
            i_sleep_deep_ua=esp_modes["deep_sleep"]["current_ma"] * 1000.0,
        )
        st.success(
            f"**Формула (2.1):** Время безубыточности перехода из `light_sleep` в `deep_sleep` "
            f"для ESP32 составляет **$t_{{be}} = {t_be:.2f}$ с**."
        )

    with c_g6:
        st.subheader("Гипотеза Г6: Априорная vs Откалиброванная модель")
        r_apr = evaluate_node_energy_budget(
            configs["apriori"],
            sf=sf_input,
            payload_bytes=payload_bytes,
            mcu_config=mcu_choice,
            cpu_freq_mhz=float(cpu_freq),
            sleep_mode=sleep_mode,
            wait_mode=wait_mode,
            tx_period_s=float(tx_period_min) * 60.0,
            battery_type=battery_type,
            **radio_overrides,
        )
        r_cal = evaluate_node_energy_budget(
            configs["calibrated"],
            sf=sf_input,
            payload_bytes=payload_bytes,
            mcu_config=mcu_choice,
            cpu_freq_mhz=float(cpu_freq),
            sleep_mode=sleep_mode,
            wait_mode=wait_mode,
            tx_period_s=float(tx_period_min) * 60.0,
            battery_type=battery_type,
            **radio_overrides,
        )

        comp_df = pd.DataFrame([
            {
                "Показатель": "Ток сна узла, мкА",
                "Априорная (даташит)": round(r_apr.i_sleep_total_ua, 2),
                "Откалиброванная (макет)": round(r_cal.i_sleep_total_ua, 2),
            },
            {
                "Показатель": "Энергия транзакции E_ц, мДж",
                "Априорная (даташит)": round(r_apr.e_tx_rx_cycle_mj, 2),
                "Откалиброванная (макет)": round(r_cal.e_tx_rx_cycle_mj, 2),
            },
            {
                "Показатель": "Средний ток I_avg, мкА",
                "Априорная (даташит)": round(r_apr.i_avg_total_ma * 1000.0, 2),
                "Откалиброванная (макет)": round(r_cal.i_avg_total_ma * 1000.0, 2),
            },
            {
                "Показатель": "Срок службы, лет",
                "Априорная (даташит)": round(r_apr.battery.lifetime_years, 2),
                "Откалиброванная (макет)": round(r_cal.battery.lifetime_years, 2),
            },
        ])
        st.dataframe(comp_df, width="stretch", hide_index=True)

        fig_g6, ax_g6 = plt.subplots(figsize=(6.2, 3.5))
        categories = ["Априорная\n(только чипы)", "Откалиброванная\n(с током платы DXL-IoT)"]
        values_years = [r_apr.battery.lifetime_years, r_cal.battery.lifetime_years]
        bars = ax_g6.bar(categories, values_years, color=["#2ca02c", "#d62728"], width=0.45)
        ax_g6.set_ylabel("Расчётный срок службы, лет")
        ax_g6.set_title(f"Завышение срока службы априорной моделью (T_tx = {tx_period_min} мин)")
        ax_g6.bar_label(bars, fmt="%.2f лет")
        ax_g6.grid(axis="y", linestyle="--", alpha=0.5)
        show_terminal_plot(fig_g6)

# ---------------------------------------------------------------------
# ВКЛАДКА 3: ПОВЕДЕНИЕ В ОЖИДАНИИ (Г5) И КЛАСС B (Г3)
# ---------------------------------------------------------------------
with tabs[2]:
    c_g5, c_g3 = st.columns(2)

    with c_g5:
        st.subheader("Гипотеза Г5: Частота МК vs Режим в паузах WAIT_RX1/RX2")
        freqs = [80, 160, 240] if mcu_choice == "K1_ESP32" else [2, 4, 8]
        e_light = [
            evaluate_node_energy_budget(
                active_cfg,
                mcu_config=mcu_choice,
                cpu_freq_mhz=f,
                wait_mode="light_sleep",
                sf=sf_input,
                **radio_overrides,
            ).e_tx_rx_cycle_mj
            for f in freqs
        ]
        e_active = [
            evaluate_node_energy_budget(
                active_cfg,
                mcu_config=mcu_choice,
                cpu_freq_mhz=f,
                wait_mode="active",
                sf=sf_input,
                **radio_overrides,
            ).e_tx_rx_cycle_mj
            for f in freqs
        ]

        fig_g5, ax_g5 = plt.subplots(figsize=(6.0, 4.0))
        x_f = np.arange(len(freqs))
        w = 0.35
        b1 = ax_g5.bar(x_f - w / 2, e_light, w, label="Ожидание: Light Sleep / Idle", color="#1f77b4")
        b2 = ax_g5.bar(x_f + w / 2, e_active, w, label="Ожидание: Активное (Active)", color="#ff7f0e")
        ax_g5.set_xticks(x_f)
        ax_g5.set_xticklabels([f"{f} МГц" for f in freqs])
        ax_g5.set_ylabel("Энергия транзакции E_ц, мДж")
        ax_g5.set_title("Влияние поведения МК в паузах приёма против тактовой частоты")
        ax_g5.legend()
        ax_g5.grid(axis="y", linestyle="--", alpha=0.5)
        show_terminal_plot(fig_g5)

    with c_g3:
        st.subheader("Гипотеза Г3: Надбавка среднего тока в Классе B")
        modes_b = [("Класс A", "A", None), ("Класс B (k=7, 128 с)", "B", 7), ("Класс B (k=4, 16 с)", "B", 4), ("Класс B (k=1, 2 с)", "B", 1)]
        i_base_list, i_addon_list = [], []
        for _, cl, k_item in modes_b:
            r_b = evaluate_node_energy_budget(
                active_cfg,
                sf=sf_input,
                device_class=cl,
                ping_periodicity_k=k_item,
                tx_period_s=float(tx_period_min) * 60.0,
                **radio_overrides,
            )
            i_base_list.append(r_b.i_avg_class_a_ma * 1000.0)
            i_addon_list.append(r_b.delta_i_class_b_ma * 1000.0)

        fig_g3, ax_g3 = plt.subplots(figsize=(6.0, 4.0))
        labels_b = [m[0] for m in modes_b]
        ax_g3.bar(labels_b, i_base_list, label="Базовый расход Класса A, мкА", color="#2ca02c")
        ax_g3.bar(labels_b, i_addon_list, bottom=i_base_list, label="Надбавка Класса B (маяки + ping), мкА", color="#d62728")
        ax_g3.set_ylabel("Средний ток потребления, мкА")
        ax_g3.set_title("Доминирование ping-слотов при малых значениях периодичности k")
        ax_g3.legend()
        ax_g3.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=12)
        show_terminal_plot(fig_g3)

# ---------------------------------------------------------------------
# ВКЛАДКА 4: ДАЛЬНОСТЬ И ЭФФЕКТИВНОСТЬ ADR (Г4)
# ---------------------------------------------------------------------
with tabs[3]:
    st.write(
        f"Uplink: порог шлюза {res.adr_info.receiver_sensitivity_dbm:.1f} дБм, "
        f"запас {res.adr_info.link_margin_db:.1f} дБ. "
        f"Downlink: пороги узла RX1 {res.rx1_sensitivity_dbm:.1f} / RX2 {res.rx2_sensitivity_dbm:.1f} дБм, "
        f"запас RX1 {res.downlink_rx1_margin_db:.1f} / RX2 {res.downlink_rx2_margin_db:.1f} дБ."
    )
    if not res.link_feasible:
        st.warning("Связь в обоих направлениях не обеспечена: проверьте запасы uplink и downlink.")
    if res.class_b_link_feasible is not None:
        st.caption(
            f"Класс B: запас приёма маяка {res.beacon_margin_db:.1f} дБ, "
            f"ping-slot {res.ping_slot_margin_db:.1f} дБ."
        )
    st.subheader("Гипотеза Г4: Зависимость выбранного SF и энергии транзакции от расстояния (ADR вкл. vs SF12 фикс.)")
    dists = active_cfg["experiment_grid"]["factors"]["distance_km_levels"]
    sf_adr_list, e_adr_list, e_fixed_list, snr_list = [], [], [], []

    for d in dists:
        r_on = evaluate_node_energy_budget(
            active_cfg,
            adr_enabled=True,
            distance_km=d,
            payload_bytes=payload_bytes,
            **radio_overrides,
        )
        r_off = evaluate_node_energy_budget(
            active_cfg,
            adr_enabled=False,
            sf=12,
            distance_km=d,
            payload_bytes=payload_bytes,
            **radio_overrides,
        )
        sf_adr_list.append(r_on.sf)
        e_adr_list.append(r_on.e_tx_rx_cycle_mj)
        e_fixed_list.append(r_off.e_tx_rx_cycle_mj)
        snr_list.append(r_on.adr_info.snr_db)

    c_adr1, c_adr2 = st.columns(2)
    with c_adr1:
        fig_adr1, ax_adr1 = plt.subplots(figsize=(6.0, 3.8))
        ax_adr1.plot(dists, e_fixed_list, "r--", linewidth=2, label="ADR выкл. (фиксированный SF12)")
        ax_adr1.plot(dists, e_adr_list, "bo-", linewidth=2, label="ADR вкл. (адаптивный выбор SF)")
        ax_adr1.set_xlabel("Расстояние до шлюза d, км")
        ax_adr1.set_ylabel("Энергия транзакции E_ц, мДж")
        ax_adr1.set_title("Экономия энергии передачи за счёт механизма ADR")
        ax_adr1.grid(True, linestyle="--", alpha=0.5)
        ax_adr1.legend()
        show_terminal_plot(fig_adr1)

    with c_adr2:
        fig_adr2, ax_adr2 = plt.subplots(figsize=(6.0, 3.8))
        ax_adr2.step(dists, sf_adr_list, where="mid", color="#9467bd", linewidth=2.5, marker="o")
        ax_adr2.set_xlabel("Расстояние до шлюза d, км")
        ax_adr2.set_ylabel("Установившийся SF")
        ax_adr2.set_yticks([7, 8, 9, 10, 11, 12])
        ax_adr2.set_title("Ступенчатое переключение SF алгоритмом ADR (формула 2.7)")
        ax_adr2.grid(True, linestyle="--", alpha=0.5)
        show_terminal_plot(fig_adr2)

# ---------------------------------------------------------------------
# ВКЛАДКА 5: АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ (Ф. 2.11) И МАТРИЦА ПФЭ 2^3 (Ф. 2.12)
# ---------------------------------------------------------------------
with tabs[4]:
    c_sens, c_pfe = st.columns(2)

    with c_sens:
        st.subheader(f"Диаграмма чувствительности (T_tx = {tx_period_min} мин)")
        sens_items = compute_sensitivity_ranking(
            active_cfg,
            tx_period_s=float(tx_period_min) * 60.0,
            evaluation_overrides={
                **radio_overrides,
                "mcu_config": mcu_choice,
                "cpu_freq_mhz": float(cpu_freq),
                "sleep_mode": sleep_mode,
                "wait_mode": wait_mode,
                "battery_type": battery_type,
            },
        )
        fig_sens, ax_sens = plt.subplots(figsize=(6.0, 4.2))
        names = [f"{it.factor_name} ({it.base_value} → {it.varied_value})" for it in reversed(sens_items)]
        vals = [it.rel_lifetime_change_pct for it in reversed(sens_items)]
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in vals]
        ax_sens.barh(names, vals, color=colors)
        ax_sens.axvline(0, color="black", linewidth=0.8)
        ax_sens.set_xlabel("Относительное изменение срока службы ΔT / T₀, %")
        ax_sens.set_title("Ранжирование факторов по влиянию на срок службы")
        ax_sens.grid(axis="x", linestyle="--", alpha=0.5)
        show_terminal_plot(fig_sens)

    with c_pfe:
        st.subheader("Полный факторный эксперимент ПФЭ 2³ (Таблица 2.6)")
        pfe_res = run_pfe_2_3_experiment(
            active_cfg,
            tx_period_s=60.0,
            evaluation_overrides={
                **radio_overrides,
                "mcu_config": mcu_choice,
                "cpu_freq_mhz": float(cpu_freq),
                "wait_mode": wait_mode,
                "battery_type": battery_type,
            },
        )
        pfe_rows = []
        for term, val in pfe_res.coefficients.items():
            row = {"Коэффициент": term, "Значение b_j": round(val, 4)}
            if pfe_res.significant_terms is not None:
                row["Значим по Стьюденту (p < 0.05)"] = "[+] Да" if pfe_res.significant_terms[term] else "[-] Нет"
            pfe_rows.append(row)
        st.dataframe(pd.DataFrame(pfe_rows), width="stretch", hide_index=True)
        if pfe_res.significant_terms is None:
            st.caption(
                "Коэффициенты рассчитаны по модели. Значимость (критерий Стьюдента) и однородность "
                "дисперсий (критерий Кохрена) проверяются только по повторным измерениям серий С1, С2 и С6."
            )
        else:
            st.caption(
                f"Критерий Кохрена $G_{{obs}} = {pfe_res.cochran_g_obs:.4f}$, "
                f"$G_{{crit}} = {pfe_res.cochran_g_critical:.4f}$ "
                f"(дисперсии однородны: **{'Да' if pfe_res.cochran_dispersions_uniform else 'Нет'}**). "
                f"Порог Стьюдента $t_{{crit}} = {pfe_res.t_critical:.3f}$, $s_b = {pfe_res.coeff_std_error_sb:.5f}$."
            )
