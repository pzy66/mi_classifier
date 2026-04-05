"""Serial-port discovery helpers shared across collection and realtime tools."""

from __future__ import annotations

import os
import re
import subprocess


_COM_PORT_PATTERN = re.compile(r"\bCOM\d+\b", flags=re.IGNORECASE)
_WINDOWS_UNAVAILABLE_STATUSES = {"error", "problem", "disconnected", "unknown", "not present", "notpresent"}


def _dedupe_keep_order(items: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = str(item).strip().upper()
        if not normalized or normalized in seen:
            continue
        ordered.append(normalized)
        seen.add(normalized)
    return ordered


def _detect_pyserial_ports() -> list[str]:
    try:
        from serial.tools import list_ports

        devices = [str(port.device).strip().upper() for port in list_ports.comports() if str(port.device).strip()]
        return _dedupe_keep_order(sorted(devices))
    except Exception:
        return []


def _parse_windows_pnp_ports(raw_output: str) -> list[tuple[str, str]]:
    ports: list[tuple[str, str]] = []
    for block in re.split(r"(?:\r?\n){2,}", str(raw_output)):
        port_match = _COM_PORT_PATTERN.search(block)
        if port_match is None:
            continue
        status = ""
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key.strip().lower() == "status":
                status = value.strip()
                break
        ports.append((port_match.group(0).upper(), status))
    return ports


def _detect_windows_pnp_ports() -> list[tuple[str, str]]:
    if os.name != "nt":
        return []
    try:
        completed = subprocess.run(
            ["pnputil", "/enum-devices", "/class", "Ports"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    return _parse_windows_pnp_ports(completed.stdout.decode("utf-8", errors="replace"))


def detect_serial_ports(*, fallback_limit: int = 20) -> list[str]:
    """Detect serial ports with Windows-specific filtering for broken COM entries."""
    pyserial_ports = _detect_pyserial_ports()

    if os.name == "nt":
        windows_ports = _detect_windows_pnp_ports()
        if windows_ports:
            status_by_port = {port: status.strip().lower() for port, status in windows_ports}
            merged: list[str] = []

            for port in pyserial_ports:
                if status_by_port.get(port, "") in _WINDOWS_UNAVAILABLE_STATUSES:
                    continue
                merged.append(port)

            for port, status in windows_ports:
                if status.strip().lower() in _WINDOWS_UNAVAILABLE_STATUSES:
                    continue
                merged.append(port)

            merged = _dedupe_keep_order(merged)
            return merged

    if pyserial_ports:
        return pyserial_ports
    return [f"COM{i}" for i in range(1, int(fallback_limit) + 1)]


def describe_serial_port(port: str) -> dict[str, object]:
    """Return a lightweight diagnostic snapshot for one serial port."""
    normalized = str(port).strip().upper()
    windows_status = ""
    if os.name == "nt" and normalized:
        for candidate_port, status in _detect_windows_pnp_ports():
            if str(candidate_port).strip().upper() == normalized:
                windows_status = str(status).strip()
                break
    return {
        "requested_port": normalized,
        "detected_ports": detect_serial_ports(),
        "windows_status": windows_status,
    }
