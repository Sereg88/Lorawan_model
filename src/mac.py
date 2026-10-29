"""
Модуль уровня MAC LoRaWAN (п. 2.6.1, 2.6.3).
Расчёт длительности окон приёма RX1 и RX2 по формуле (2.2),
а также расчёт временных параметров и токовой надбавки класса B по формуле (2.5).
"""

from dataclasses import dataclass
from src.phy import calc_symbol_duration_ms, calc_toa_ms


@dataclass(frozen=True)
class ClassATimings:
    """Длительности фаз транзакционного цикла класса A, мс."""
    sf_tx: int
    sf_rx1: int
    sf_rx2: int
    tx_ms: float
    wait_rx1_ms: float
    rx1_ms: float
    wait_rx2_ms: float
    rx2_ms: float


@dataclass(frozen=True)
class ClassBMetrics:
    """Параметры периодической активности класса B (формула 2.5)."""
    periodicity_k: int          # Параметр периодичности k in {0..7}
    slots_per_beacon: int       # Число ping-слотов за период маяка: 2^(7 - k)
    ping_period_s: float        # Интервал между ping-слотами T_ping = 0,96 * 2^k, с
    beacon_period_s: float      # Период маяка T_b = 128 с
    q_beacon_mc: float          # Заряд на приём одного маяка Q_b, мКл
    q_ping_mc: float            # Заряд на один ping-слот Q_p, мКл
    delta_i_class_b_ma: float   # Надбавка среднего тока в классе B, мА


def calc_rx_window_duration_ms(
    sf: int,
    n_sym: int = 8,
    bw_khz: float = 125.0,
) -> float:
    """
    Формула (2.2): длительность окна приёма при отсутствии преамбулы
    T_rx = N_sym * T_sym(SF), мс.
    """
    if n_sym <= 0:
        raise ValueError(f"Число символов окна приёма должно быть положительным, получено: {n_sym}")
    return n_sym * calc_symbol_duration_ms(sf=sf, bw_khz=bw_khz)


def get_rx1_sf(sf_tx: int, rx1_dr_offset: int = 0) -> int:
    """
    Определяет SF для первого окна приёма RX1 в плане EU433.
    При RX1DROffset = 0 равен SF восходящего сообщения.
    """
    if not (7 <= sf_tx <= 12):
        raise ValueError(f"Некорректный SF восходящего кадра: {sf_tx}")
    return min(12, max(7, sf_tx + rx1_dr_offset))


def calc_class_a_timings(
    sf_tx: int,
    payload_bytes: int,
    bw_khz: float = 125.0,
    coding_rate: int = 1,
    preamble_symbols: int = 8,
    lorawan_header_bytes: int = 13,
    n_sym_rx: int = 8,
    rx1_delay_ms: float = 1000.0,
    rx2_delay_ms: float = 2000.0,
    rx1_dr_offset: int = 0,
    rx2_sf: int = 12,
) -> ClassATimings:
    """
    Вычисляет длительности радиофаз цикла класса A (TX, WAIT_RX1, RX1, WAIT_RX2, RX2).
    Задержки RECEIVE_DELAY1 (rx1_delay_ms) и RECEIVE_DELAY2 (rx2_delay_ms) отсчитываются
    от конца передачи, поэтому пауза перед RX2 равна RECEIVE_DELAY2 - RECEIVE_DELAY1 - T_rx1
    (ср. таблицу 1.3: «1000 - T_RX1»).
    Важное свойство для гипотезы Г1: длительность окна RX2 определяется rx2_sf=12 (DR0)
    и не зависит от SF восходящего сообщения.
    """
    tx_ms = calc_toa_ms(
        sf=sf_tx,
        payload_bytes=payload_bytes,
        bw_khz=bw_khz,
        coding_rate=coding_rate,
        preamble_symbols=preamble_symbols,
        lorawan_header_bytes=lorawan_header_bytes,
    )
    sf_rx1 = get_rx1_sf(sf_tx, rx1_dr_offset)
    rx1_ms = calc_rx_window_duration_ms(sf=sf_rx1, n_sym=n_sym_rx, bw_khz=bw_khz)
    rx2_ms = calc_rx_window_duration_ms(sf=rx2_sf, n_sym=n_sym_rx, bw_khz=bw_khz)
    wait_rx2_ms = rx2_delay_ms - rx1_delay_ms - rx1_ms
    if wait_rx2_ms < 0:
        raise ValueError(
            f"Окно RX1 ({rx1_ms:.1f} мс) не укладывается между RECEIVE_DELAY1 и RECEIVE_DELAY2."
        )

    return ClassATimings(
        sf_tx=sf_tx,
        sf_rx1=sf_rx1,
        sf_rx2=rx2_sf,
        tx_ms=tx_ms,
        wait_rx1_ms=rx1_delay_ms,
        rx1_ms=rx1_ms,
        wait_rx2_ms=wait_rx2_ms,
        rx2_ms=rx2_ms,
    )


def calc_class_b_current_addon(
    periodicity_k: int,
    q_beacon_mc: float,
    q_ping_mc: float,
    beacon_period_s: float = 128.0,
    beacon_window_s: float = 122.88,
) -> ClassBMetrics:
    """
    Формула (2.5): расчёт дополнительной составляющей среднего тока в классе B:
    Delta_I_B = (Q_b + 2^(7 - k) * Q_p) / T_b.

    Параметры:
        periodicity_k: Параметр периодичности ping-слотов k (0..7).
                       При k=7 -> 1 слот за период маяка; k=4 -> 8 слотов (15,36 с); k=1 -> 64 слота (1,92 с).
        q_beacon_mc: Суммарный заряд (радио + МК) на пробуждение и приём маяка Q_b, мКл.
        q_ping_mc: Суммарный заряд (радио + МК) на открытие одного ping-слота Q_p, мКл.
        beacon_period_s: Период следования маяков T_b = 128 с.
        beacon_window_s: Окно ping-слотов внутри периода маяка (4096 слотов по 30 мс = 122,88 с),
                         по нему считается интервал между ping-слотами 0,96 * 2^k с.
    """
    if not (0 <= periodicity_k <= 7):
        raise ValueError(f"Периодичность ping-слотов k должна быть в диапазоне 0..7, получено: {periodicity_k}")
    if beacon_period_s <= 0:
        raise ValueError("Период маяка T_b должен быть положительным.")

    slots_per_beacon = 2 ** (7 - periodicity_k)
    ping_period_s = beacon_window_s / slots_per_beacon
    # Так как Q в мКл (мА * с), деление на секунды сразу даёт ток в мА
    delta_i_ma = (q_beacon_mc + slots_per_beacon * q_ping_mc) / beacon_period_s

    return ClassBMetrics(
        periodicity_k=periodicity_k,
        slots_per_beacon=slots_per_beacon,
        ping_period_s=ping_period_s,
        beacon_period_s=beacon_period_s,
        q_beacon_mc=q_beacon_mc,
        q_ping_mc=q_ping_mc,
        delta_i_class_b_ma=delta_i_ma,
    )