import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = PROJECT_ROOT / "code" / "shared"
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from src.serial_ports import _parse_windows_pnp_ports, detect_serial_ports


class SerialPortDetectionTests(TestCase):
    def test_parse_windows_pnp_ports_extracts_port_status_pairs(self) -> None:
        raw_output = """
Microsoft PnP Utility

Instance ID:                FTDIBUS\\VID_0403+PID_6015+D30DVCDYA\\0000
Device Description:         USB Serial Port (COM4)
Class Name:                 Ports
Status:                     Disconnected

Instance ID:                ROOT\\PORTS\\0000
Device Description:         Broken Port (COM3)
Class Name:                 Ports
Status:                     Problem
"""
        self.assertEqual(
            _parse_windows_pnp_ports(raw_output),
            [("COM4", "Disconnected"), ("COM3", "Problem")],
        )

    @patch("src.serial_ports.os.name", "nt")
    @patch("src.serial_ports._detect_windows_pnp_ports", return_value=[("COM4", "Disconnected"), ("COM3", "Problem")])
    @patch("src.serial_ports._detect_pyserial_ports", return_value=["COM3", "COM4"])
    def test_detect_serial_ports_filters_unavailable_ports_from_windows_inventory(
        self,
        _mock_pyserial,
        _mock_windows_ports,
    ) -> None:
        self.assertEqual(detect_serial_ports(), [])

    @patch("src.serial_ports.os.name", "nt")
    @patch("src.serial_ports._detect_windows_pnp_ports", return_value=[("COM4", "OK")])
    @patch("src.serial_ports._detect_pyserial_ports", return_value=["COM4"])
    def test_detect_serial_ports_keeps_healthy_windows_ports(
        self,
        _mock_pyserial,
        _mock_windows_ports,
    ) -> None:
        self.assertEqual(detect_serial_ports(), ["COM4"])
