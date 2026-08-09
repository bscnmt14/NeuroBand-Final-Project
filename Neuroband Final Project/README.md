# NeuroBand

NeuroBand is a three-sensor surface-EMG system for personal hand-gesture
recognition, realtime computer control, and an interactive target game. The
software receives uMyo packets over a serial bridge, reconstructs the eight EMG
values in each packet as consecutive time samples, fuses three fixed forearm
locations, and applies a user-specific hierarchical classifier.

## Hardware Mapping

| uMyo ID | Forearm location |
| --- | --- |
| `B0DAC7E9` | Ventral forearm |
| `ED7A78C8` | Dorsal forearm |
| `37ED348F` | Inner forearm side |

All three sensors are required for classification. The active classes are
`at_rest`, `fist`, `like`, `open_hand`, `pinch`, `pointing`,
`wrist_extension`, and `wrist_flexion`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The application is developed for Windows because realtime cursor control uses
Windows input APIs. The EMG viewer and offline training modules remain ordinary
Python/PySide6 code.

## Run

Realtime recognition, personal calibration, replay testing, and mouse control:

```powershell
python NeuroBand_files\run_realtime_gui.py
```

Standalone target game:

```powershell
python NeuroBand_files\run_neuroband_shooter.py
```

Personal training is normally launched from the GUI. It can also be started from
the command line:

```powershell
python NeuroBand_files\personal_stage_training.py `
  --session-dir Data\calibration_sessions\SESSION_NAME `
  --mode full
```

## Signal And Model Pipeline

1. uMyo packets are decoded by the retained OEM parser.
2. `emg_0..emg_7` are flattened as eight sequential EMG samples per packet.
3. Each sensor is processed with a fourth-order 35-500 Hz Butterworth band-pass
   filter and a 50 Hz notch filter.
4. Synchronized windows from the three fixed sensors are fused.
5. Model-specific normalization and feature extraction are applied.
6. A hierarchical model first separates rest from activity and then classifies
   the active gesture.
7. Confidence rejection and temporal decision logic stabilize realtime output.

See [Code Architecture](docs/CODE_ARCHITECTURE.md) for the module-level design.

## Repository Layout

```text
NeuroBand_files/   Developed application, training, and evaluation code
OEM_files/         Minimal original uMyo parser dependencies
pictures/          GUI, calibration, gesture, and game assets
Data/              Local runtime data location; recordings are ignored by Git
docs/              Architecture documentation and diagram source
```

## Data And Models

Personal EMG recordings, trained `.pkl` models, replay outputs, diagnostic logs,
and experiment results are intentionally not included. They can contain personal
biometric data and are generated locally under `Data/calibration_sessions`.

## Main Entry Points

- `run_realtime_gui.py`: realtime GUI and calibration workflow.
- `run_neuroband_shooter.py`: standalone game.
- `personal_stage_training.py`: personal Full Grid/Fast training engine.
- `replay_evaluation.py`: chronological replay and decision comparison.
- `training_data_pipeline.py`: offline feature construction and experiments.
- `model_training.py`: signal-processing and hierarchical-model primitives.

