"""
Единый перечень состояний узла LoRaWAN, цепей питания и загрузчик конфигурации YAML.
Обеспечивает единые границы учёта энергии между аналитической моделью и обработкой осциллограмм (п. 2.2, 2.6.3).
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Union
import yaml


class PowerCircuit(str, Enum):
    """Цепи питания измерительного стенда (п. 2.2, рисунок 2.2)."""
    RADIO = "radio"          # Цепь 1: Радиомодуль (DRF1278F / EBYTE)
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
    RX2 = "RX2"              # Второе окно приёма (фиксировано DR0 / SF12 в плане EU433)
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
                voltage_v=self.voltage_v,
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
        """Возвращает заряд состояния (по одной цепи или суммарно по обеим), мКл."""
        if state not in self.states:
            return 0.0
        if circuit is not None:
            return self.states[state][circuit].charge_mc
        return sum(m.charge_mc for m in self.states[state].values())

    def get_state_energy_mj(self, state: NodeState, circuit: PowerCircuit | None = None) -> float:
        """Возвращает энергию состояния (по одной цепи или суммарно по обеим), мДж."""
        return self.voltage_v * self.get_state_charge_mc(state, circuit)

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