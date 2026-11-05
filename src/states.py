"""
Единый перечень состояний узла LoRaWAN, цепей питания и загрузчик конфигурации YAML.
Обеспечивает единые границы учёта энергии между аналитической моделью и обработкой осциллограмм (п. 2.2, 2.6.3).
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import yaml


class PowerCircuit(str, Enum):
    """Цепи питания измерительного стенда (п. 2.2, рисунок 2.2)."""
    RADIO = "radio"          # Цепь 1: радиомодуль, напряжение может отличаться от питания МК
    MCU_BOARD = "mcu_board"  # Цепь 2: Плата DXL-IoT (ESP32 / ATmega328P) + датчик BME280


class NodeState(str, Enum):
    """
    Регистрируемые состояния узла (рисунок 2.3):
    а) Транзакционный цикл класса A: WU -> TX -> WAIT_RX1 -> RX1 -> WAIT_RX2 -> RX2 -> PP -> SLEEP
    б) Дополнительные состояния опроса датчика и класса B: SENSOR, BEACON_RX, PING_SLOT
    """
    WU = "WU"                # Пробуждение микроконтроллера и подготовка кадра
    TX = "TX"                # Передача восходящего кадра (Uplink)
    WAIT_RX1 = "WAIT_RX1"    # Пауза ожидания первого окна приёма (обычно 1 с)
    RX1 = "RX1"              # Первое окно приёма (на SF восходящего кадра при RX1DROffset=0)
    WAIT_RX2 = "WAIT_RX2"    # Пауза ожидания второго окна приёма (обычно 1 с)
    RX2 = "RX2"              # Второе окно приёма (задаётся выбранным частотным планом)
    PP = "PP"                # Завершающая обработка (Post-processing) перед уходом в сон
    SLEEP = "SLEEP"          # Режим сна между активными фазами
    SENSOR = "SENSOR"        # Принудительное измерение датчиком BME280 (forced mode)
    BEACON_RX = "BEACON_RX"  # Приём синхромаяка в классе B (каждые 128 с)
    PING_SLOT = "PING_SLOT"  # Открытие ping-слота в классе B


# Упорядоченная последовательность активных состояний транзакции класса A (рисунок 2.3, а)
CLASS_A_ACTIVE_SEQUENCE: tuple[NodeState, ...] = (
    NodeState.WU,
    NodeState.TX,
    NodeState.WAIT_RX1,
    NodeState.RX1,
    NodeState.WAIT_RX2,
    NodeState.RX2,
    NodeState.PP,
)


@dataclass
class StateMetrics:
    """Показатели отдельного состояния для конкретной цепи питания (Таблица 2.2)."""
    state: NodeState
    circuit: PowerCircuit
    duration_ms: float       # Длительность состояния t_i, мс
    current_ma: float        # Ток потребления I_i, мА
    voltage_v: float = 3.3   # Напряжение питания U, В

    @property
    def duration_s(self) -> float:
        """Длительность состояния в секундах."""
        return self.duration_ms / 1000.0

    @property
    def charge_mc(self) -> float:
        """Заряд состояния Q_i = I_i * t_i, мКл (милликулоны, т.к. мА * с = мКл)."""
        return self.current_ma * self.duration_s

    @property
    def energy_mj(self) -> float:
        """Энергия состояния E_i = U * Q_i, мДж."""
        return self.voltage_v * self.charge_mc


@dataclass
class CycleProfile:
    """Профиль транзакционного цикла с разбивкой по двум цепям и состояниям."""
    voltage_v: float = 3.3
    radio_voltage_v: float | None = None
    states: Dict[NodeState, Dict[PowerCircuit, StateMetrics]] = field(default_factory=dict)

    def add_state(
        self,
        state: NodeState,
        duration_ms: float,
        radio_current_ma: float,
        mcu_current_ma: float,
    ) -> None:
        """Добавляет или обновляет состояние сразу для обеих цепей питания."""
        self.states[state] = {
            PowerCircuit.RADIO: StateMetrics(
                state=state,
                circuit=PowerCircuit.RADIO,
                duration_ms=duration_ms,
                current_ma=radio_current_ma,
                voltage_v=self.voltage_v if self.radio_voltage_v is None else self.radio_voltage_v,
            ),
            PowerCircuit.MCU_BOARD: StateMetrics(
                state=state,
                circuit=PowerCircuit.MCU_BOARD,
                duration_ms=duration_ms,
                current_ma=mcu_current_ma,
                voltage_v=self.voltage_v,
            ),
        }

    def get_state_charge_mc(self, state: NodeState, circuit: PowerCircuit | None = None) -> float:
        """Заряд цепи — физический; сумма цепей приведена к voltage_v без потерь питания."""
        if state not in self.states:
            return 0.0
        if circuit is not None:
            return self.states[state][circuit].charge_mc
        # Суммарный заряд разных шин приводится к опорному напряжению платы.
        return sum(m.energy_mj for m in self.states[state].values()) / self.voltage_v

    def get_state_energy_mj(self, state: NodeState, circuit: PowerCircuit | None = None) -> float:
        """Возвращает энергию состояния (по одной цепи или суммарно по обеим), мДж."""
        if state not in self.states:
            return 0.0
        if circuit is not None:
            return self.states[state][circuit].energy_mj
        return sum(m.energy_mj for m in self.states[state].values())

    def get_active_duration_ms(self) -> float:
        """Суммарная длительность активных состояний транзакции класса A (t_act), мс."""
        return sum(
            self.states[st][PowerCircuit.RADIO].duration_ms
            for st in CLASS_A_ACTIVE_SEQUENCE
            if st in self.states
        )


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Загружает конфигурационный YAML-файл модели (априорный или откалиброванный)."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Конфигурационный файл не найден: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_band_plan_names(config: Dict[str, Any]) -> List[str]:
    """Возвращает список доступных частотных планов из конфига."""
    band_plans = config.get("band_plans")
    if band_plans:
        return [str(name) for name in band_plans.keys()]
    legacy_region = config.get("system", {}).get("region", "default")
    return [str(legacy_region)]


def get_supported_band_plans(config: Dict[str, Any], radio_module: str) -> List[str]:
    """
    Возвращает список частотных планов, совместимых с радиомодулем.
    Если у модуля список не задан, считаем допустимыми все band plan'ы конфига.
    """
    radio_modules = config.get("radio_modules", {})
    if radio_module not in radio_modules:
        available = ", ".join(sorted(map(str, radio_modules.keys())))
        raise ValueError(
            f"Неизвестный радиомодуль '{radio_module}'. Доступные варианты: {available}."
        )

    all_plans = get_band_plan_names(config)
    supported = radio_modules[radio_module].get("supported_band_plans", all_plans)
    supported_list = [str(plan) for plan in supported]
    invalid = [plan for plan in supported_list if plan not in all_plans]
    if invalid:
        raise ValueError(
            f"Радиомодуль '{radio_module}' ссылается на неизвестные частотные планы: {', '.join(invalid)}."
        )
    return supported_list


def resolve_band_plan(config: Dict[str, Any], band_plan: str | None = None) -> Tuple[str, Dict[str, Any]]:
    """
    Возвращает имя активного band plan'а и его конфигурацию.
    Поддерживает новую схему `band_plans` и старую схему с верхнеуровневыми system/class_b/channel_adr.
    """
    band_plans = config.get("band_plans")
    if band_plans:
        baseline = config.get("experiment_grid", {}).get("baseline", {})
        selected_name = str(
            baseline.get("band_plan", next(iter(band_plans))) if band_plan is None else band_plan
        )
        if selected_name not in band_plans:
            available = ", ".join(sorted(map(str, band_plans.keys())))
            raise ValueError(
                f"Неизвестный частотный план '{selected_name}'. Доступные варианты: {available}."
            )
        return selected_name, band_plans[selected_name]

    legacy_system = config.get("system")
    legacy_class_b = config.get("class_b")
    legacy_channel_adr = config.get("channel_adr")
    if legacy_system is None or legacy_class_b is None or legacy_channel_adr is None:
        raise KeyError("В конфигурации отсутствуют band_plans и legacy-секции system/class_b/channel_adr.")

    selected_name = str(legacy_system.get("region", "legacy") if band_plan is None else band_plan)
    return selected_name, {
        "label": selected_name,
        "system": legacy_system,
        "class_b": legacy_class_b,
        "channel_adr": legacy_channel_adr,
    }
