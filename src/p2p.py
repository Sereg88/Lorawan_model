"""Оценка канала E220 P2P → прикладной мост → оконечный узел LoRaWAN.

Это модель и контракт полезной нагрузки, не драйвер E220 и не прошивка моста.
Штатный E220 никогда не получает кадры MAC LoRaWAN через UART.
"""

from copy import deepcopy
from dataclasses import dataclass
import math
import struct
import zlib

from src.battery import BatteryLifetimeResult, evaluate_battery_from_config
from src.channel_adr import calc_link_budget
from src.node import (
    EnergyBudgetResult, calc_mcu_active_current_ma, evaluate_node_energy_budget,
    get_sleep_wakeup_cost,
)
from src.states import NodeState, PowerCircuit


_HEADER = struct.Struct("!2sBHIH")  # magic, version, sensor_id, sequence, payload length
ENVELOPE_BYTES = _HEADER.size + 4  # CRC32 прикладного сообщения


@dataclass(frozen=True)
class P2PMessage:
    sensor_id: int
    sequence: int
    payload: bytes


def encode_p2p_message(sensor_id: int, sequence: int, payload: bytes) -> bytes:
    """Тот же контейнер передаётся мостом в FRMPayload LoRaWAN (отдельный FPort)."""
    if not 0 <= sensor_id <= 65535 or not 0 <= sequence <= 0xFFFFFFFF:
        raise ValueError("sensor_id должен быть uint16, sequence — uint32.")
    if not isinstance(payload, bytes) or not 1 <= len(payload) <= 65535:
        raise ValueError("Нужны непустые данные bytes длиной до 65535 байт.")
    body = _HEADER.pack(b"LP", 1, sensor_id, sequence, len(payload)) + payload
    return body + struct.pack("!I", zlib.crc32(body))


def decode_p2p_message(frame: bytes) -> P2PMessage:
    if not isinstance(frame, bytes) or len(frame) <= ENVELOPE_BYTES:
        raise ValueError("Неполное сообщение P2P.")
    magic, version, sensor_id, sequence, size = _HEADER.unpack_from(frame)
    if magic != b"LP" or version != 1 or len(frame) != ENVELOPE_BYTES + size:
        raise ValueError("Неверный формат сообщения P2P.")
    if zlib.crc32(frame[:-4]) != struct.unpack("!I", frame[-4:])[0]:
        raise ValueError("Ошибка CRC32 сообщения P2P.")
    return P2PMessage(sensor_id, sequence, frame[_HEADER.size:-4])


@dataclass(frozen=True)
class BridgeUplink:
    fport: int
    sensor_id: int
    sequence: int
    payload: bytes  # Передать в API стека LoRaWAN; это не готовый MAC-кадр


def prepare_lorawan_uplink(frame: bytes, fport: int = 20) -> BridgeUplink:
    """Проверка принятого P2P-контейнера перед отправкой приложением моста.

    Ограничение 51 байт выбрано для всего моделируемого набора SF. Дедупликация,
    управление очередью, радиодрайверы и сессия LoRaWAN — обязанности прошивки.
    """
    if not isinstance(fport, int) or not 1 <= fport <= 223:
        raise ValueError("Нужен прикладной FPort 1..223.")
    message = decode_p2p_message(frame)
    if len(frame) > 51:
        raise ValueError("Контейнер превышает 51 байт; требуется фрагментация или другая настройка DR.")
    return BridgeUplink(fport, message.sensor_id, message.sequence, frame)


@dataclass(frozen=True)
class P2PPhase:
    name: str
    duration_ms: float
    radio_energy_mj: float  # Энергия от опорной шины с учётом КПД радио
    mcu_energy_mj: float

    @property
    def energy_mj(self) -> float:
        return self.radio_energy_mj + self.mcu_energy_mj


@dataclass(frozen=True)
class P2PBridgeResult:
    channel: int
    frequency_hz: int
    tx_power_dbm: int
    air_rate_bps: int
    payload_bytes: int
    forwarded_bytes: int
    airtime_ms: float
    airtime_source: str
    tx_period_s: float
    reference_voltage_v: float
    parameter_profile: str
    sensor_phases: tuple[P2PPhase, ...]
    sensor_cycle_energy_mj: float
    sensor_avg_current_ma: float
    sensor_battery: BatteryLifetimeResult
    bridge_lorawan: EnergyBudgetResult
    bridge_cycle_energy_mj: float
    bridge_p2p_rx_energy_mj: float
    bridge_processing_energy_mj: float
    bridge_avg_current_ma: float
    bridge_battery: BatteryLifetimeResult
    p2p_rssi_dbm: float
    p2p_sensitivity_dbm: float
    p2p_margin_db: float
    end_to_end_link_feasible: bool
    warnings: tuple[str, ...]
    forward_fport: int


def evaluate_p2p_bridge(
    config: dict, *, channel: int = 23, tx_power_dbm: int = 22,
    air_rate_bps: int = 2400, payload_bytes: int = 12, tx_period_s: float = 600,
    distance_km: float = 0.5, measured_airtime_ms: float | None = None,
    assumed_radio_overhead_bytes: int = 0, parameter_profile: str | None = None,
    sensitivity_dbm: float | None = None,
    mcu_config: str = "K1_ESP32", cpu_freq_mhz: float | None = None,
    sleep_mode: str | None = None, battery_type: str = "ideal",
    bridge_radio_module: str = "DRF1278F", bridge_band_plan: str = "EU433",
    bridge_sf: int = 9, bridge_distance_km: float = 0.5,
    bridge_mcu_config: str = "K1_ESP32",
    bridge_cpu_freq_mhz: float | None = None,
    bridge_sleep_mode: str | None = None, bridge_wait_mode: str = "light_sleep",
    bridge_device_class: str = "A", bridge_ping_periodicity_k: int | None = None,
    bridge_adr_enabled: bool = False,
    bridge_receiver_sensitivity_dbm: float | None = None,
    bridge_gateway_sensitivity_dbm: float | None = None,
) -> P2PBridgeResult:
    """Один сенсор, один контейнер/период, мост постоянно слушает вторым E220.

    Нет ACK, повторов, WOR, пакетирования нескольких сенсоров и автономного join.
    Мост пересоздаёт сообщение на прикладном уровне и использует существующую
    LoRaWAN-сессию. Контейнер содержит sensor_id/sequence/CRC32, но не шифрование.
    """
    options = config["p2p_links"]["EBYTE"]
    profile_key = parameter_profile or options["parameter_profile"]
    if profile_key not in options["parameter_profiles"]:
        raise ValueError("Неизвестный набор параметров E220 P2P.")
    radio = {**config["radio_modules"]["EBYTE"], **options["parameter_profiles"][profile_key]}
    if isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel < radio["channel_count"]:
        raise ValueError("Канал E220 должен быть целым числом 0..83.")
    if tx_power_dbm not in radio["tx_power_levels_dbm"]:
        raise ValueError("Мощность E220 должна быть 10, 13, 17 или 22 дБм.")
    if air_rate_bps not in options["air_rate_levels_bps"]:
        raise ValueError("Неподдерживаемая скорость E220.")
    if not isinstance(payload_bytes, int) or not 1 <= payload_bytes <= 65535:
        raise ValueError("Размер данных сенсора должен быть положительным целым числом.")
    if not isinstance(assumed_radio_overhead_bytes, int) or assumed_radio_overhead_bytes < 0:
        raise ValueError("Накладные байты должны быть неотрицательным целым числом.")
    for val in (tx_period_s, distance_km, bridge_distance_km):
        if not math.isfinite(val) or val <= 0:
            raise ValueError("Период и расстояния должны быть положительными конечными числами.")
    wire_bytes = payload_bytes + ENVELOPE_BYTES
    uplink = prepare_lorawan_uplink(encode_p2p_message(1, 0, bytes(payload_bytes)), options["forward_fport"])
    if wire_bytes > options["subpacket_bytes"]:
        raise ValueError("Контейнер превышает размер подпакета E220; фрагментация не моделируется.")
    # Консервативный общий предел для реализованного подмножества планов/SF7..12.
    if wire_bytes > options["lorawan_forward_payload_limit_bytes"]:
        raise ValueError("Контейнер превышает 51 байт LoRaWAN; уменьшите данные или реализуйте фрагментацию.")
    if measured_airtime_ms is not None and (not math.isfinite(measured_airtime_ms) or measured_airtime_ms <= 0):
        raise ValueError("Измеренное время передачи должно быть положительным конечным числом.")
    airtime_ms = (float(measured_airtime_ms) if measured_airtime_ms is not None else
                  8000.0 * (wire_bytes + assumed_radio_overhead_bytes) / air_rate_bps)
    warnings = [
        "Один сенсор → E220 P2P → мост с отдельным радиомодулем LoRaWAN. Это оценка архитектуры, не готовая прошивка.",
        "Время E220 по паспортной скорости — грубая оценка без известного PHY-формата; для проверки энергии задайте измеренное время TX."
        if measured_airtime_ms is None else "Использовано заданное измеренное время TX E220.",
        "Мост постоянно слушает E220; ACK, повторы, WOR, коллизии и новые OTAA-подключения не учитываются.",
        "UART и TX считаются последовательными; в UART-фазе ток радио принят равным RX, переходы требуют измерения.",
    ]
    if tx_power_dbm != 22:
        warnings.append("Ток TX на выбранной мощности неизвестен; использован ориентир для +22 дБм.")
    if air_rate_bps != 2400:
        warnings.append("Для этой скорости нет паспортной чувствительности; введённый порог RX является допущением.")
    mcu = config["mcu_configs"][mcu_config]
    frequency = float(cpu_freq_mhz if cpu_freq_mhz is not None else mcu["base_freq_mhz"])
    if frequency not in mcu["active_current_by_freq_ma"]:
        raise ValueError("Частота МК должна соответствовать выбранной платформе.")
    mode = sleep_mode or ("deep_sleep" if mcu_config == "K1_ESP32" else "power_down")
    mode_cfg = mcu["sleep_modes"][mode]
    voltage = float(config["system"]["supply_voltage_v"])
    radio_v = float(radio["supply_voltage_v"])
    eta = float(radio.get("power_conversion_efficiency", 1.0))
    if voltage <= 0 or not 0 < eta <= 1 or not radio["supply_voltage_range_v"][0] <= radio_v <= radio["supply_voltage_range_v"][1]:
        raise ValueError("Некорректное питание E220 или КПД.")
    sensor = config["sensor"]
    active = calc_mcu_active_current_ma(mcu, frequency)
    idle = float(mode_cfg["current_ma"]) + float(mcu.get("board_quiescent_current_ma", 0)) + float(mcu.get("co_mcu_hibernation_ma", 0))
    sensor_sleep = float(sensor["sleep_current_ma"])
    wake_ms, wake_q = get_sleep_wakeup_cost(mode_cfg, voltage)
    prep_ms = float(mcu["wu_duration_base_ms"]) * mcu["base_freq_mhz"] / frequency
    pp_ms = float(mcu["pp_duration_base_ms"]) * mcu["base_freq_mhz"] / frequency
    uart_ms = 1000.0 * wire_bytes * 10 / options["uart_baud_bps"] + options["uart_flush_ms"]
    phases = []

    def add(name, duration_ms, radio_ma, mcu_ma, extra_mcu_mj=0):
        phases.append(P2PPhase(name, duration_ms, radio_v * radio_ma * duration_ms / 1000 / eta,
                               voltage * mcu_ma * duration_ms / 1000 + extra_mcu_mj))

    add("WAKE_PREP", wake_ms + prep_ms, radio["sleep_current_ma"],
        0, voltage * wake_q + voltage * (active + sensor_sleep) * prep_ms / 1000)
    add("SENSOR", sensor["measurement_duration_ms"], radio["sleep_current_ma"], active + sensor["active_current_ma"])
    add("UART", uart_ms, radio["rx_current_ma"], active + sensor_sleep)
    add("P2P_TX", airtime_ms, radio["tx_current_ma"], active + sensor_sleep)
    add("POST", pp_ms, radio["sleep_current_ma"], active + sensor_sleep)
    sleep_ms = tx_period_s * 1000 - sum(p.duration_ms for p in phases)
    if sleep_ms < 0:
        raise ValueError("Период короче полного цикла E220 P2P.")
    add("SLEEP", sleep_ms, radio["sleep_current_ma"], idle + sensor_sleep)
    sensor_energy = sum(p.energy_mj for p in phases)
    sensor_i = sensor_energy / voltage / tx_period_s

    # У моста свой МК и отдельное радио LoRaWAN. BME280 на мосту не предполагается.
    bridge_cfg = deepcopy(config)
    bridge_cfg["sensor"].update(active_current_ma=0, sleep_current_ma=0, measurement_duration_ms=0)
    bmcu = config["mcu_configs"][bridge_mcu_config]
    bridge_frequency = float(bridge_cpu_freq_mhz if bridge_cpu_freq_mhz is not None else bmcu["base_freq_mhz"])
    if bridge_frequency not in bmcu["active_current_by_freq_ma"]:
        raise ValueError("Частота МК моста должна соответствовать выбранной платформе.")
    if bridge_device_class not in ("A", "B"):
        raise ValueError("Класс LoRaWAN моста должен быть A или B.")
    if bridge_wait_mode not in ("light_sleep", "active"):
        raise ValueError("Неизвестный режим ожидания RX моста.")
    forward = evaluate_node_energy_budget(
        bridge_cfg, radio_module=bridge_radio_module, band_plan=bridge_band_plan,
        mcu_config=bridge_mcu_config, cpu_freq_mhz=bridge_frequency,
        sf=bridge_sf, adr_enabled=bridge_adr_enabled,
        payload_bytes=wire_bytes, sensor_freq_hz=0, tx_period_s=tx_period_s,
        device_class=bridge_device_class, ping_periodicity_k=bridge_ping_periodicity_k,
        wait_mode=bridge_wait_mode, distance_km=bridge_distance_km, battery_type=battery_type,
        sleep_mode=bridge_sleep_mode or ("deep_sleep" if bridge_mcu_config == "K1_ESP32" else "power_down"),
        receiver_sensitivity_dbm=bridge_receiver_sensitivity_dbm,
        gateway_sensitivity_dbm=bridge_gateway_sensitivity_dbm,
    )
    proc_ms = float(options["bridge_processing_ms"])
    if not math.isfinite(proc_ms) or proc_ms < 0 or proc_ms > forward.sleep_duration_s * 1000:
        raise ValueError("Обработка на мосту не помещается в период.")
    bridge_active = calc_mcu_active_current_ma(bmcu, forward.cpu_freq_mhz)
    bridge_sleep = forward.profile.states[NodeState.SLEEP][PowerCircuit.MCU_BOARD].current_ma
    proc_energy = max(0, bridge_active - bridge_sleep) * forward.profile.voltage_v * proc_ms / 1000
    bridge_rx_energy = radio_v * radio["rx_current_ma"] * tx_period_s / eta
    bridge_energy = forward.e_cycle_total_mj + bridge_rx_energy + proc_energy
    bridge_voltage = forward.profile.voltage_v
    bridge_i = bridge_energy / bridge_voltage / tx_period_s

    frequency_hz = int(radio["channel_base_hz"] + channel * radio["channel_step_hz"])
    channel_cfg = options["channel"]
    lb = calc_link_budget(
        distance_km=distance_km, tx_power_dbm=tx_power_dbm,
        antenna_gain_db=float(channel_cfg["antenna_gain_db"]),
        ref_distance_km=float(channel_cfg["ref_distance_km"]),
        ref_loss_db=float(channel_cfg["ref_loss_db"]) + 20 * math.log10(frequency_hz / channel_cfg["reference_frequency_hz"]),
        path_loss_exponent=float(channel_cfg["path_loss_exponent"]),
    )
    threshold = float(radio["sensitivity_dbm"] if sensitivity_dbm is None else sensitivity_dbm)
    if not math.isfinite(threshold):
        raise ValueError("Порог чувствительности должен быть конечным числом.")
    warnings.extend(forward.model_warnings)
    warnings.append("Контейнер ID/sequence/CRC32 не обеспечивает шифрование и аутентификацию P2P; они требуют отдельной реализации.")
    return P2PBridgeResult(
        channel, frequency_hz, tx_power_dbm, air_rate_bps, payload_bytes, wire_bytes,
        airtime_ms, "measured" if measured_airtime_ms is not None else "estimated",
        tx_period_s, voltage, profile_key, tuple(phases), sensor_energy, sensor_i,
        evaluate_battery_from_config(config, battery_type, sensor_i), forward,
        bridge_energy, bridge_rx_energy, proc_energy, bridge_i,
        evaluate_battery_from_config(config, battery_type, bridge_i),
        lb.rssi_dbm, threshold, lb.rssi_dbm - threshold,
        lb.rssi_dbm >= threshold and forward.link_feasible, tuple(warnings), uplink.fport,
    )
