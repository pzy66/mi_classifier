import os
import sys
import threading
import time
import unittest
from unittest import mock
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_ROOT = PROJECT_ROOT / "code" / "collection"
SHARED_ROOT = PROJECT_ROOT / "code" / "shared"
if str(COLLECTION_ROOT) not in sys.path:
    sys.path.insert(0, str(COLLECTION_ROOT))
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from mi_data_collector import MIDataCollectorWindow
from src.mi_collection import SessionSettings, make_event


class MIDataCollectorSessionFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _build_window_with_fake_session(self) -> MIDataCollectorWindow:
        window = MIDataCollectorWindow()
        window.show_error = lambda message: None
        window.log = lambda message: None
        window.worker_thread = object()
        window.worker = None
        window.session_running = True
        window.waiting_for_save = False
        window.session_paused = False
        window.capture_on_stop = True
        window.event_log = []
        window.trial_records = []

        def fake_record_event(
            event_name: str,
            *,
            trial_id: int | None = None,
            class_name: str | None = None,
            run_index: int | None = None,
            run_trial_index: int | None = None,
            block_index: int | None = None,
            prompt_index: int | None = None,
            command_duration_sec: float | None = None,
            execution_success: int | bool | None = None,
        ) -> None:
            window.event_log.append(
                make_event(
                    event_name,
                    trial_id=trial_id,
                    class_name=class_name,
                    run_index=run_index,
                    run_trial_index=run_trial_index,
                    block_index=block_index,
                    prompt_index=prompt_index,
                    command_duration_sec=command_duration_sec,
                    execution_success=execution_success,
                )
            )

        window.record_event = fake_record_event
        return window

    def _build_settings(self) -> SessionSettings:
        return SessionSettings(
            subject_id="sub-test",
            session_id="20260403_120000",
            output_root=str(PROJECT_ROOT / "runtime" / "test_output"),
            board_id=0,
            serial_port="COM4",
            channel_names=["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
            channel_positions=list(range(8)),
            trials_per_class=1,
            baseline_sec=1.0,
            cue_sec=2.0,
            imagery_sec=1.0,
            iti_sec=1.0,
            random_seed=1234,
            run_count=2,
            continuous_block_count=0,
            continuous_block_sec=0.0,
        )

    def _pump(self, seconds: float = 0.05) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            self.app.processEvents()
            time.sleep(0.01)

    @staticmethod
    def _visible_control_button_names(window: MIDataCollectorWindow) -> set[str]:
        buttons = [
            window.connect_button,
            window.start_button,
            window.start_mi_only_button,
            window.pause_button,
            window.bad_trial_button,
            window.stop_button,
            window.disconnect_button,
        ]
        return {button.objectName() for button in buttons if button is not None and not button.isHidden()}

    def test_finish_session_does_not_duplicate_terminal_end_markers(self) -> None:
        phase_cases = [
            ("idle_block", "idle_block_end", {"block_index": 1}, {"idle_block_index": 1}),
            ("idle_prepare", "idle_prepare_end", {"block_index": 1}, {"idle_prepare_block_index": 1}),
            ("continuous", "continuous_block_end", {"block_index": 2}, {"continuous_block_index": 2}),
        ]

        for phase_name, end_event, end_kwargs, state_kwargs in phase_cases:
            with self.subTest(phase=phase_name):
                window = self._build_window_with_fake_session()
                try:
                    window.current_phase = phase_name
                    window.current_continuous_prompt = None
                    for attr_name, attr_value in state_kwargs.items():
                        setattr(window, attr_name, attr_value)
                    window.event_log.append(make_event(end_event, **end_kwargs))

                    window.finish_session_and_request_save(manual_stop=False)

                    end_count = sum(1 for event in window.event_log if str(event.get("event_name", "")) == end_event)
                    session_end_count = sum(
                        1 for event in window.event_log if str(event.get("event_name", "")) == "session_end"
                    )
                    self.assertEqual(end_count, 1, f"{phase_name} appended a duplicate terminal end marker")
                    self.assertEqual(session_end_count, 1, f"{phase_name} failed to append session_end exactly once")
                finally:
                    window.waiting_for_save = False
                    window.worker_thread = None
                    window.close()

    def test_config_section_defaults_to_device(self) -> None:
        window = MIDataCollectorWindow()
        try:
            self.assertIsNotNone(window.config_section_combo)
            self.assertIsNotNone(window.config_stack)
            self.assertEqual(window.config_section_combo.currentText(), "设备")
            self.assertEqual(window.config_stack.currentIndex(), window.config_section_combo.currentIndex())
        finally:
            window.close()

    def test_disconnected_control_panel_only_shows_connect(self) -> None:
        window = MIDataCollectorWindow()
        try:
            window.device_info = None
            window.worker_thread = None
            window.session_running = False
            window.waiting_for_save = False
            window.update_button_states()

            self.assertEqual(self._visible_control_button_names(window), {"btnConnect"})
            self.assertEqual(window.control_layout_columns, 1)
        finally:
            window.close()

    def test_connected_idle_control_panel_shows_start_and_disconnect(self) -> None:
        window = MIDataCollectorWindow()
        try:
            window.device_info = {
                "sampling_rate": 250.0,
                "channel_names": ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
                "selected_rows": list(range(8)),
            }
            window.worker_thread = None
            window.session_running = False
            window.waiting_for_save = False
            window.update_button_states()

            self.assertEqual(
                self._visible_control_button_names(window),
                {"btnStart", "btnStartMiOnly", "btnDisconnect"},
            )
            self.assertEqual(window.control_layout_columns, 1)
        finally:
            window.worker_thread = None
            window.close()

    def test_preview_mode_switch_is_queued_and_locks_controls_until_completion(self) -> None:
        class FakeWorker:
            def supports_impedance_mode(self) -> bool:
                return True

        window = MIDataCollectorWindow()
        window.show_error = lambda message: None
        window.log = lambda message: None
        window.device_info = {
            "sampling_rate": 250.0,
            "channel_names": ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
            "selected_rows": list(range(8)),
        }
        window.worker = FakeWorker()
        window.preview_mode = "EEG"
        emitted: list[tuple[str, int, bool]] = []
        window.preview_mode_switch_requested.connect(lambda mode, channel, reset: emitted.append((mode, channel, reset)))
        try:
            window.update_button_states()
            self.assertTrue(window.start_button.isEnabled())
            self.assertTrue(window.disconnect_button.isEnabled())

            accepted = window._switch_preview_mode("IMP", channel=3)

            self.assertTrue(accepted)
            self.assertTrue(window.preview_mode_switch_pending)
            self.assertEqual(emitted, [("IMP", 3, False)])
            self.assertFalse(window.start_button.isEnabled())
            self.assertFalse(window.disconnect_button.isEnabled())
            self.assertFalse(window.preview_to_imp_button.isEnabled())

            window.on_preview_mode_switch_finished(
                {
                    "ok": True,
                    "message": "",
                    "target_mode": "IMP",
                    "target_channel": 3,
                    "reset_default": False,
                }
            )

            self.assertFalse(window.preview_mode_switch_pending)
            self.assertEqual(window.preview_mode, "IMP")
            self.assertEqual(window.preview_impedance_channel, 3)
            self.assertTrue(window.start_button.isEnabled())
            self.assertTrue(window.disconnect_button.isEnabled())
        finally:
            window.close()

    def test_start_session_waits_for_eeg_switch_completion(self) -> None:
        class FakeWorker:
            def supports_impedance_mode(self) -> bool:
                return True

        window = MIDataCollectorWindow()
        window.show_error = lambda message: None
        window.log = lambda message: None
        window.device_info = {
            "sampling_rate": 250.0,
            "channel_names": ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
            "selected_rows": list(range(8)),
        }
        window.worker = FakeWorker()
        window.preview_mode = "IMP"
        window.preview_impedance_channel = 4
        window.collect_settings = self._build_settings
        emitted: list[tuple[str, int, bool]] = []
        begin_calls: list[tuple[SessionSettings, list[list[str]]]] = []
        window.preview_mode_switch_requested.connect(lambda mode, channel, reset: emitted.append((mode, channel, reset)))
        window._begin_session_with_settings = lambda settings, sequence_by_run: begin_calls.append((settings, sequence_by_run))
        try:
            window.start_session()

            self.assertFalse(begin_calls)
            self.assertFalse(window.session_running)
            self.assertIsNotNone(window.pending_session_start_context)
            self.assertTrue(window.preview_mode_switch_pending)
            self.assertEqual(emitted, [("EEG", 4, True)])

            window.on_preview_mode_switch_finished(
                {
                    "ok": True,
                    "message": "",
                    "target_mode": "EEG",
                    "target_channel": 4,
                    "reset_default": True,
                }
            )

            self.assertFalse(window.preview_mode_switch_pending)
            self.assertIsNone(window.pending_session_start_context)
            self.assertEqual(len(begin_calls), 1)
            self.assertEqual(begin_calls[0][0].session_id, "20260403_120000")
            self.assertEqual(len(begin_calls[0][1]), 2)
        finally:
            window.close()

    def test_start_mi_only_session_applies_protocol_overrides_before_begin(self) -> None:
        class FakeWorker:
            def supports_impedance_mode(self) -> bool:
                return False

        window = MIDataCollectorWindow()
        window.show_error = lambda message: None
        window.log = lambda message: None
        window.device_info = {
            "sampling_rate": 250.0,
            "channel_names": ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
            "selected_rows": list(range(8)),
        }
        window.worker = FakeWorker()
        begin_calls: list[tuple[SessionSettings, list[list[str]]]] = []
        window.collect_settings = self._build_settings
        window._begin_session_with_settings = lambda settings, sequence_by_run: begin_calls.append((settings, sequence_by_run))
        try:
            window.start_mi_only_session()

            self.assertEqual(len(begin_calls), 1)
            settings, sequence_by_run = begin_calls[0]
            self.assertEqual(settings.protocol_mode, "mi_only")
            self.assertEqual(settings.practice_sec, 0.0)
            self.assertEqual(settings.calibration_open_sec, 0.0)
            self.assertEqual(settings.calibration_closed_sec, 0.0)
            self.assertEqual(settings.idle_block_count, 0)
            self.assertEqual(settings.idle_prepare_block_count, 0)
            self.assertEqual(settings.continuous_block_count, 0)
            self.assertEqual(settings.run_rest_sec, 0.0)
            self.assertEqual(settings.long_run_rest_every, 0)
            self.assertEqual(len(sequence_by_run), 2)
        finally:
            window.close()

    def test_start_session_validation_failure_does_not_queue_hardware_switch(self) -> None:
        class FakeWorker:
            def supports_impedance_mode(self) -> bool:
                return True

        window = MIDataCollectorWindow()
        errors: list[str] = []
        window.show_error = errors.append
        window.log = lambda message: None
        window.device_info = {
            "sampling_rate": 250.0,
            "channel_names": ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
            "selected_rows": list(range(8)),
        }
        window.worker = FakeWorker()
        window.preview_mode = "IMP"
        window.collect_settings = lambda: (_ for _ in ()).throw(ValueError("bad settings"))
        emitted: list[tuple[str, int, bool]] = []
        window.preview_mode_switch_requested.connect(lambda mode, channel, reset: emitted.append((mode, channel, reset)))
        try:
            window.start_session()

            self.assertEqual(errors, ["bad settings"])
            self.assertFalse(window.preview_mode_switch_pending)
            self.assertIsNone(window.pending_session_start_context)
            self.assertEqual(emitted, [])
        finally:
            window.close()

    def test_baseline_to_cue_transition_triggers_class_voice_prompt(self) -> None:
        window = self._build_window_with_fake_session()
        spoken_classes: list[str | None] = []
        window.current_settings = self._build_settings()
        window.current_phase = "baseline"
        window.phase_deadline = time.perf_counter() - 0.1
        window.current_trial = mock.Mock()
        window.current_trial.trial_id = 1
        window.current_trial.class_name = "left_hand"
        window.current_trial.run_index = 1
        window.current_trial.run_trial_index = 1
        window.current_trial.display_name = "左手"
        window._speak_cue_prompt = lambda class_name: spoken_classes.append(class_name)
        try:
            window.on_phase_tick()

            self.assertEqual(window.current_phase, "cue")
            self.assertEqual(spoken_classes, ["left_hand"])
        finally:
            window.waiting_for_save = False
            window.session_running = False
            window.worker_thread = None
            window.close()

    def test_session_data_ready_saves_in_background_and_updates_ui_on_completion(self) -> None:
        window = MIDataCollectorWindow()
        logs: list[str] = []
        errors: list[str] = []
        window.log = logs.append
        window.show_error = errors.append
        window.capture_on_stop = True
        window.waiting_for_save = True
        window.current_settings = self._build_settings()
        window.event_log = [make_event("session_start")]
        window.trial_records = []
        window.use_separate_participant_screen = False

        payload = {
            "brainflow_data": np.zeros((10, 32), dtype=np.float32),
            "sampling_rate": 250.0,
            "selected_rows": list(range(8)),
            "marker_row": 8,
            "timestamp_row": None,
            "package_num_row": None,
            "board_descr": {},
        }

        save_started = threading.Event()
        release_save = threading.Event()

        def fake_save_mi_session(**kwargs):
            del kwargs
            save_started.set()
            release_save.wait(timeout=1.0)
            return {
                "trial_count": 1,
                "session_dir": str(PROJECT_ROOT / "runtime" / "async_save"),
                "fif_path": str(PROJECT_ROOT / "runtime" / "async_save" / "run_raw.fif"),
                "board_data_path": "",
                "segments_csv_path": "",
                "mi_epochs_path": "",
                "gate_epochs_path": "",
                "artifact_epochs_path": "",
                "continuous_path": "",
                "manifest_csv_path": "",
                "save_index": 1,
                "run_stem": "sub-test_ses-20260403_120000_run-001_tpc-01_n-001_ok-001",
            }

        try:
            started_at = time.perf_counter()
            with mock.patch("mi_data_collector.save_mi_session", side_effect=fake_save_mi_session):
                window.on_session_data_ready(payload)
                elapsed = time.perf_counter() - started_at

                self.assertLess(elapsed, 0.1)
                self.assertTrue(window.waiting_for_save)
                self.assertIsNotNone(window.save_thread)
                self.assertTrue(save_started.wait(timeout=0.5))

                release_save.set()
                deadline = time.time() + 1.5
                while window.save_thread is not None and time.time() < deadline:
                    self._pump(0.05)

                self.assertFalse(window.waiting_for_save)
                self.assertIsNone(window.save_thread)
                self.assertIsNone(window.save_worker)
                self.assertEqual(errors, [])
                self.assertIn("当前任务：数据已保存到", window.current_label.text())
                self.assertTrue(any("开始后台写盘" in item for item in logs))
        finally:
            release_save.set()
            if window.save_thread is not None:
                deadline = time.time() + 1.0
                while window.save_thread is not None and time.time() < deadline:
                    self._pump(0.05)
            window.waiting_for_save = False
            window.close()


if __name__ == "__main__":
    unittest.main()
