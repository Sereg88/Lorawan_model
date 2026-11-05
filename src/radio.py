"""Паспортные параметры радиомодулей и явно заданные приближения модели."""

import math
from typing import Any

from src.channel_adr import DEFAULT_SNR_THRESHOLDS_DB


def get_radio_compatibility_note(radio: dict[str, Any]) -> str:
    """Пояснение необязательно в YAML; статус совместимости остаётся значимым."""
    if radio.get("compatibility_note"):
        return str(radio["compatibility_note"])
    if radio.get("lorawan_support") == "unsupported_stock_firmware":
        return "Штатная прошивка радиомодуля несовместима с расчётом LoRaWAN."
    if radio.get("lorawan_support") == "requires_custom_firmware":
        return "Расчёт LoRaWAN предполагает использование подходящей прошивки радиомодуля."
    return "Для радиомодуля нет совместимых частотных планов LoRaWAN в конфигурации."


def sensitivity_at_sf(
    sensitivity_dbm: float,
    sf: int,
    reference_sf: int,
    bandwidth_khz: float = 125.0,
    reference_bandwidth_khz: float = 125.0,
) -> float:
    """Оценка порога RX из опорного значения: ΔSNR(SF) + 10 lg(BW/BW_ref).

    Это расчётная зависимость при постоянном NF, не паспортная таблица по SF.
    Опорные SF/BW должны быть известны либо явно приняты как допущение.
    """
    if sf not in DEFAULT_SNR_THRESHOLDS_DB or reference_sf not in DEFAULT_SNR_THRESHOLDS_DB:
        raise ValueError("Для расчёта чувствительности SF должен быть в диапазоне 7..12.")
    if not math.isfinite(sensitivity_dbm) or bandwidth_khz <= 0 or reference_bandwidth_khz <= 0:
        raise ValueError("Некорректные параметры чувствительности или полосы приёмника.")
    return (
        sensitivity_dbm
        + DEFAULT_SNR_THRESHOLDS_DB[sf]
        - DEFAULT_SNR_THRESHOLDS_DB[reference_sf]
        + 10.0 * math.log10(bandwidth_khz / reference_bandwidth_khz)
    )


def module_sensitivity_dbm(
    radio: dict[str, Any], sf: int, bandwidth_khz: float = 125.0,
    override_dbm: float | None = None,
) -> float:
    """Порог приёма узла; override меняет чувствительность в опорном режиме модуля."""
    ref_sf = radio.get("sensitivity_reference_sf")
    if ref_sf is None:
        raise ValueError("Даташит не связывает чувствительность с SF/BW; нужен измеренный профиль RX.")
    return sensitivity_at_sf(
        float(radio["sensitivity_dbm"] if override_dbm is None else override_dbm),
        sf, int(ref_sf), bandwidth_khz,
        float(radio["sensitivity_reference_bandwidth_khz"]),
    )


def validate_radio_band_plan(radio: dict[str, Any], plan: dict[str, Any]) -> None:
    """Проверяет диапазон, сетку каналов и необходимые SF, включая RX2 и класс B."""
    if radio.get("lorawan_support") == "unsupported_stock_firmware":
        raise ValueError(get_radio_compatibility_note(radio))
    system = plan["system"]
    frequencies = [system["frequency_hz"], system.get("rx2_frequency_hz", system["frequency_hz"])]
    frequencies += system.get("default_uplink_channels_hz", [])
    frequencies += [plan["class_b"].get("beacon_frequency_hz", frequencies[0]),
                    plan["class_b"].get("ping_slot_frequency_hz", frequencies[0])]
    lo, hi = radio["frequency_range_mhz"]
    for hz in frequencies:
        if not lo * 1e6 <= hz <= hi * 1e6:
            raise ValueError(f"Частота {hz / 1e6:g} МГц вне диапазона радиомодуля {lo}..{hi} МГц.")
        step = radio.get("channel_step_hz")
        if step and abs((hz - radio["channel_base_hz"]) / step - round((hz - radio["channel_base_hz"]) / step)) > 1e-6:
            raise ValueError(f"Частота {hz / 1e6:g} МГц недоступна на сетке каналов модуля.")
    sfs = radio.get("sf_by_bandwidth_khz", {}).get(int(system["bandwidth_khz"]))
    if sfs is not None and any(sf not in sfs for sf in [system["rx2_sf"], plan["class_b"]["beacon_sf"]]):
        raise ValueError("Радиомодуль не поддерживает SF приёмных окон выбранного плана.")
