"""
Интерактивный веб-дашборд (Streamlit) для аналитической модели энергопотребления узла LoRaWAN.
Визуализирует энергетический бюджет, профили состояний, проверку гипотез Г1–Г6,
анализ чувствительности (торнадо) и результаты ПФЭ 2^3.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.states import CLASS_A_ACTIVE_SEQUENCE, PowerCircuit, load_config
from src.phy import generate_table_2_4
from src.channel_adr import resolve_adr
from src.node import calc_sleep_breakeven_time_s, evaluate_node_energy_budget
from src.experiments import compute_sensitivity_ranking, run_pfe_2_3_experiment

# Настройка страницы
st.set_page_config(
    page_title="LoRaWAN Energy Model (DXL-IoT)",
    page_icon="📡",
    layout="wide",
)

ROOT_DIR = Path(__file__).resolve().parent


@st.cache_data
def get_configs():
    return {
        "apriori": load_config(ROOT_DIR / "config" / "apriori_params.yaml"),
        "calibrated": load_config(ROOT_DIR / "config" / "calibrated_params.yaml"),
    }


configs = get_configs()

# =====================================================================
# БОКОВАЯ ПАНЕЛЬ УПРАВЛЕНИЯ ПАРАМЕТРАМИ
# =====================================================================
st.sidebar.title("⚙️ Параметры узла")

model_choice = st.sidebar.radio(
    "Тип модели (Гипотеза Г6):",
    options=["calibrated", "apriori"],
    format_func=lambda x: "Откалиброванная (стенд DXL-IoT)" if x == "calibrated" else "Априорная (по даташитам)",
)
active_cfg = configs[model_choice]

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

radio_module = st.sidebar.selectbox("Радиомодуль (SX1278, EU433):", options=["DRF1278F", "EBYTE"])
wait_mode = st.sidebar.radio(
    "Ожидание окон RX1/RX2 (Г5):",
    options=["light_sleep", "active"],
    format_func=lambda x: "Неглубокий сон (Light/Idle)" if x == "light_sleep" else "Активное ожидание (Active)",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📻 Радиоинтерфейс и трафик")

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
    radio_module=radio_module,
)

# =====================================================================
# ГЛАВНЫЙ ЭКРАН: СВОДНЫЕ КАРТОЧКИ МЕТРИК
# =====================================================================
st.title("📡 Аналитическая модель энергопотребления узла LoRaWAN")
st.caption(
    f"Конфигурация: **{mcu_choice}** ({cpu_freq} МГц) | Радиомодуль: **{radio_module}** | "
    f"Фактический **SF{res.sf}** ($P_{{tx}} = {res.tx_power_dbm:.0f}$ дБм) | "
    f"Класс **{res.device_class}**"
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Срок службы", f"{res.battery.lifetime_years:.2f} лет", f"{res.battery.lifetime_days:.0f} сут.")
c2.metric("Средний ток I_avg", f"{res.i_avg_total_ma * 1000.0:.1f} мкА", f"{res.i_avg_total_ma:.4f} мА")
c3.metric("Энергия транзакции", f"{res.e_tx_rx_cycle_mj:.2f} мДж", f"t_act = {res.active_tx_rx_duration_ms:.0f} мс")
c4.metric("Суточный бюджет E_сут", f"{res.e_day_total_j:.2f} Дж/сут", f"Сон: {res.share_sleep_pct:.1f}%")
c5.metric("Ток сна (суммарный)", f"{res.i_sleep_total_ua:.1f} мкА", f"Ожидание RX: {res.share_wait_windows_pct:.1f}%")

st.markdown("---")

# =====================================================================
# ВКЛАДКИ ИССЛЕДОВАНИЯ И ПРОВЕРКИ ГИПОТЕЗ Г1–Г6
# =====================================================================
tabs = st.tabs([
    "📊 Профиль цикла и Г1",
    "🌙 Сон и период передачи (Г2, Г6)",
    "⚡ Ожидание RX и Класс B (Г3, Г5)",
    "📶 Дальность и ADR (Г4)",
    "🌪️ Чувствительность и ПФЭ 2³",
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
        st.dataframe(df_states, use_container_width=True, hide_index=True)

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
        st.pyplot(fig1)

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
                radio_module=radio_module,
            )
            toa_list.append(r_s.profile.states[CLASS_A_ACTIVE_SEQUENCE[1]][PowerCircuit.RADIO].duration_ms)
            e_tx_rx_list.append(r_s.e_tx_rx_cycle_mj)
            rx2_share_list.append(r_s.share_rx2_in_tx_rx_pct)

        toa_ratio = toa_list[-1] / toa_list[0]
        e_ratio = e_tx_rx_list[-1] / e_tx_rx_list[0]
        st.info(
            f"**Вывод по Г1 (при L = {payload_bytes} байт):** переход от SF7 к SF12 увеличивает "
            f"время передачи в **{toa_ratio:.1f} раза**, тогда как полная энергия радиотранзакции "
            f"возрастает лишь в **{e_ratio:.1f} раза** (за счёт фиксированного окна RX2 на DR0 и константных затрат на пробуждение/ожидание)."
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
        st.pyplot(fig2)

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
        st.pyplot(fig_g2)

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
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        fig_g6, ax_g6 = plt.subplots(figsize=(6.2, 3.5))
        categories = ["Априорная\n(только чипы)", "Откалиброванная\n(с током платы DXL-IoT)"]
        values_years = [r_apr.battery.lifetime_years, r_cal.battery.lifetime_years]
        bars = ax_g6.bar(categories, values_years, color=["#2ca02c", "#d62728"], width=0.45)
        ax_g6.set_ylabel("Расчётный срок службы, лет")
        ax_g6.set_title(f"Завышение срока службы априорной моделью (T_tx = {tx_period_min} мин)")
        ax_g6.bar_label(bars, fmt="%.2f лет")
        ax_g6.grid(axis="y", linestyle="--", alpha=0.5)
        st.pyplot(fig_g6)

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
                active_cfg, mcu_config=mcu_choice, cpu_freq_mhz=f, wait_mode="light_sleep", sf=sf_input
            ).e_tx_rx_cycle_mj
            for f in freqs
        ]
        e_active = [
            evaluate_node_energy_budget(
                active_cfg, mcu_config=mcu_choice, cpu_freq_mhz=f, wait_mode="active", sf=sf_input
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
        st.pyplot(fig_g5)

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
        st.pyplot(fig_g3)

# ---------------------------------------------------------------------
# ВКЛАДКА 4: ДАЛЬНОСТЬ И ЭФФЕКТИВНОСТЬ ADR (Г4)
# ---------------------------------------------------------------------
with tabs[3]:
    st.subheader("Гипотеза Г4: Зависимость выбранного SF и энергии транзакции от расстояния (ADR вкл. vs SF12 фикс.)")
    dists = active_cfg["experiment_grid"]["factors"]["distance_km_levels"]
    sf_adr_list, e_adr_list, e_fixed_list, snr_list = [], [], [], []

    for d in dists:
        r_on = evaluate_node_energy_budget(active_cfg, adr_enabled=True, distance_km=d, payload_bytes=payload_bytes)
        r_off = evaluate_node_energy_budget(active_cfg, adr_enabled=False, sf=12, distance_km=d, payload_bytes=payload_bytes)
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
        st.pyplot(fig_adr1)

    with c_adr2:
        fig_adr2, ax_adr2 = plt.subplots(figsize=(6.0, 3.8))
        ax_adr2.step(dists, sf_adr_list, where="mid", color="#9467bd", linewidth=2.5, marker="o")
        ax_adr2.set_xlabel("Расстояние до шлюза d, км")
        ax_adr2.set_ylabel("Установившийся SF")
        ax_adr2.set_yticks([7, 8, 9, 10, 11, 12])
        ax_adr2.set_title("Ступенчатое переключение SF алгоритмом ADR (формула 2.7)")
        ax_adr2.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_adr2)

# ---------------------------------------------------------------------
# ВКЛАДКА 5: АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ (Ф. 2.11) И МАТРИЦА ПФЭ 2^3 (Ф. 2.12)
# ---------------------------------------------------------------------
with tabs[4]:
    c_sens, c_pfe = st.columns(2)

    with c_sens:
        st.subheader(f"Диаграмма чувствительности (T_tx = {tx_period_min} мин)")
        sens_items = compute_sensitivity_ranking(active_cfg, tx_period_s=float(tx_period_min) * 60.0)
        fig_sens, ax_sens = plt.subplots(figsize=(6.0, 4.2))
        names = [f"{it.factor_name} ({it.base_value} → {it.varied_value})" for it in reversed(sens_items)]
        vals = [it.rel_lifetime_change_pct for it in reversed(sens_items)]
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in vals]
        ax_sens.barh(names, vals, color=colors)
        ax_sens.axvline(0, color="black", linewidth=0.8)
        ax_sens.set_xlabel("Относительное изменение срока службы ΔT / T₀, %")
        ax_sens.set_title("Ранжирование факторов по влиянию на срок службы")
        ax_sens.grid(axis="x", linestyle="--", alpha=0.5)
        st.pyplot(fig_sens)

    with c_pfe:
        st.subheader("Полный факторный эксперимент ПФЭ 2³ (Таблица 2.6)")
        pfe_res = run_pfe_2_3_experiment(active_cfg, tx_period_s=60.0)
        pfe_rows = []
        for term, val in pfe_res.coefficients.items():
            pfe_rows.append({
                "Коэффициент": term,
                "Значение b_j": round(val, 4),
                "Значим по Стьюденту (p < 0.05)": "✅ Да" if pfe_res.significant_terms[term] else "❌ Нет",
            })
        st.dataframe(pd.DataFrame(pfe_rows), use_container_width=True, hide_index=True)
        st.caption(
            f"Критерий Кохрена $G_{{obs}} = {pfe_res.cochran_g_obs:.4f} \\le 0.5157$ "
            f"(дисперсии однородны: **{'Да' if pfe_res.cochran_dispersions_uniform else 'Нет'}**). "
            f"Порог Стьюдента $t_{{crit}} = {pfe_res.t_critical:.3f}$, $s_b = {pfe_res.coeff_std_error_sb:.5f}$."
        )