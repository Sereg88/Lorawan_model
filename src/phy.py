"""
Модуль физического уровня LoRa (п. 2.6.1, 2.6.3).
Расчёт длительности символа, числа символов полезной нагрузки и времени в эфире (Time-on-Air, ToA)
по формулам Semtech SX1276/77/78/79, а также генерация Таблицы 2.4.
"""

from dataclasses import dataclass
import math
from typing import Dict, List


@dataclass(frozen=True)
class LoRaPhyBreakdown:
    """Детализация расчёта времени передачи кадра LoRa."""
    sf: int
    bw_khz: float
    payload_bytes: int
    total_phy_bytes: int
    coding_rate: int
    low_data_rate_optimize: int  # DE: 1 при T_sym >= 16 мс, иначе 0
    t_sym_ms: float              # Длительность одного символа, мс
    t_preamble_ms: float         # Длительность преамбулы, мс
    n_payload_symbols: int       # Число символов поля данных и заголовка
    t_payload_ms: float          # Длительность поля данных, мс
    toa_ms: float                # Полное время в эфире (Time-on-Air), мс


def calc_symbol_duration_ms(sf: int, bw_khz: float = 125.0) -> float:
    """
    Вычисляет длительность символа LoRa T_sym = (2^SF) / BW, мс.
    При BW в кГц отношение (2^SF) / bw_khz сразу даёт результат в миллисекундах.
    """
    if not (7 <= sf <= 12):
        raise ValueError(f"Коэффициент расширения спектра SF должен быть в диапазоне 7..12, получено: {sf}")
    if bw_khz <= 0:
        raise ValueError(f"Полоса пропускания должна быть положительной, получено: {bw_khz}")
    return (2 ** sf) / float(bw_khz)


def is_low_data_rate_optimized(sf: int, bw_khz: float = 125.0) -> int:
    """
    Определяет необходимость включения оптимизации низких скоростей (DE = 1),
    если длительность символа T_sym >= 16.0 мс (для BW=125 кГц это SF11 и SF12).
    """
    t_sym_ms = calc_symbol_duration_ms(sf, bw_khz)
    return 1 if t_sym_ms >= 16.0 else 0


def calc_lora_toa_breakdown(
    sf: int,
    payload_bytes: int,
    bw_khz: float = 125.0,
    coding_rate: int = 1,
    preamble_symbols: int = 8,
    lorawan_header_bytes: int = 13,
    explicit_header: bool = True,
    crc_enabled: bool = True,
    de_override: int | None = None,
) -> LoRaPhyBreakdown:
    """
    Вычисляет время передачи (Time-on-Air) восходящего сообщения LoRaWAN по формулам (1.2)–(1.5).

    Параметры:
        sf: Коэффициент расширения спектра (7..12).
        payload_bytes: Размер полезной нагрузки приложения L_pl, байт (в Таблице 2.4: 4, 12, 24, 51).
        bw_khz: Полоса пропускания, кГц (по умолчанию 125 кГц).
        coding_rate: Индекс скорости кода CR (1 -> 4/5, 2 -> 4/6, 3 -> 4/7, 4 -> 4/8).
        preamble_symbols: Число символов преамбулы N_pre (по умолчанию 8).
        lorawan_header_bytes: Служебные поля MAC-кадра LoRaWAN (MHDR + FHDR + FPort + MIC = 13 байт).
        explicit_header: Наличие физического заголовка PHDR (True -> H = 0, False -> H = 1).
        crc_enabled: Наличие CRC на физическом уровне (True -> CRC = 1, False -> CRC = 0).
        de_override: Принудительное задание флага DE (0 или 1); если None, определяется автоматически.
    """
    if payload_bytes < 0:
        raise ValueError("Размер полезной нагрузки не может быть отрицательным.")
    if coding_rate not in (1, 2, 3, 4):
        raise ValueError("Параметр coding_rate должен принимать значения 1 (4/5), 2 (4/6), 3 (4/7) или 4 (4/8).")

    t_sym_ms = calc_symbol_duration_ms(sf, bw_khz)
    de = is_low_data_rate_optimized(sf, bw_khz) if de_override is None else int(bool(de_override))
    h = 0 if explicit_header else 1
    crc = 1 if crc_enabled else 0

    # Полный размер кадра физического уровня PL = L_pl + 13 байт служебных полей LoRaWAN
    pl = payload_bytes + lorawan_header_bytes

    # Длительность преамбулы: T_pre = (N_pre + 4.25) * T_sym
    t_preamble_ms = (preamble_symbols + 4.25) * t_sym_ms

    # Число символов нагрузки: N_payload = 8 + max(ceil((8*PL - 4*SF + 28 + 16*CRC - 20*H) / (4*(SF - 2*DE))) * (CR + 4), 0)
    numerator = 8 * pl - 4 * sf + 28 + 16 * crc - 20 * h
    denominator = 4 * (sf - 2 * de)
    blocks = math.ceil(numerator / denominator)
    n_payload_symbols = 8 + max(blocks * (coding_rate + 4), 0)

    t_payload_ms = n_payload_symbols * t_sym_ms
    toa_ms = t_preamble_ms + t_payload_ms

    return LoRaPhyBreakdown(
        sf=sf,
        bw_khz=bw_khz,
        payload_bytes=payload_bytes,
        total_phy_bytes=pl,
        coding_rate=coding_rate,
        low_data_rate_optimize=de,
        t_sym_ms=t_sym_ms,
        t_preamble_ms=t_preamble_ms,
        n_payload_symbols=n_payload_symbols,
        t_payload_ms=t_payload_ms,
        toa_ms=toa_ms,
    )


def calc_toa_ms(
    sf: int,
    payload_bytes: int,
    bw_khz: float = 125.0,
    coding_rate: int = 1,
    preamble_symbols: int = 8,
    lorawan_header_bytes: int = 13,
    explicit_header: bool = True,
    crc_enabled: bool = True,
) -> float:
    """Короткая обёртка, возвращающая только время передачи кадра ToA в мс."""
    return calc_lora_toa_breakdown(
        sf=sf,
        payload_bytes=payload_bytes,
        bw_khz=bw_khz,
        coding_rate=coding_rate,
        preamble_symbols=preamble_symbols,
        lorawan_header_bytes=lorawan_header_bytes,
        explicit_header=explicit_header,
        crc_enabled=crc_enabled,
    ).toa_ms


def generate_table_2_4(
    sf_list: List[int] = [7, 8, 9, 10, 11, 12],
    payload_list: List[int] = [4, 12, 24, 51],
) -> Dict[int, Dict[int, float]]:
    """
    Генерирует словарь значений времени передачи (мс) для Таблицы 2.4 отчёта:
    результат[sf][payload_bytes] = toa_ms.
    """
    table: Dict[int, Dict[int, float]] = {}
    for sf in sf_list:
        table[sf] = {}
        for pl in payload_list:
            table[sf][pl] = calc_toa_ms(sf=sf, payload_bytes=pl)
    return table