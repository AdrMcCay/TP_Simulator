# Thomson Parabola Simulator (TP Simulator)

A Python-based simulation and visualisation tool for modelling ion trajectories in a **Thomson Parabola (TP) spectrometer**. Includes an interactive GUI for configuring fields, geometry, and ion species, with detector visualisation.

---

## Overview

The Thomson Parabola spectrometer is widely used in **laser–plasma physics** to measure ion energy spectra and charge-to-mass ratios.

This simulator models ion motion through:

- Magnetic field region (charge-dependent deflection)
- Electric field region (energy-dependent deflection)
- Drift region (spatial separation onto the detector)

The output reproduces the characteristic **parabolic traces** observed in experiments.

---

## Features

- **PyQt5 GUI**
  - Interactive parameter control
  - Load/save configurations

- **Multi-species ion simulation**
  - Arbitrary mass, charge state, and energy range
  - Energy spread sampling

- **Detector modelling**
  - MCP (circular detector with adjustable radius)
  - IP (rectangular detector – extendable)
  - Adjustable Zero Point

- **Visualisation**
  - 2D detector hit maps
  - Optional 3D trajectories
  - Particle animation

- **Custom configurations**
  - JSON-based settings
  - Example configs included

- **Magnetic field calibration tool**
  - Separate script for tuning B-field

---

## Repository Structure

```
TP_simulator/
│
├── TPsimulator.py          # Main GUI + simulation engine
├── Bfield_calibrator.py    # Magnetic field calibration
├── settings/
│   └── Apollon_2025/
│       ├── TP_0.json
│       └── TP_10.json
```

---

## Installation

### Requirements

- Python 3.8+
- PyQt5
- NumPy
- Matplotlib

### Install dependencies

```bash
pip install pyqt5 numpy matplotlib
```

---

## Usage

Run the simulator:

```bash
python TPsimulator.py
```

### Workflow

1. Set spectrometer parameters:
   - Magnetic field (`B`)
   - Plate voltage (`V_plate`)
   - Geometry (lengths, distances)

2. Define ion species:
   - Mass (amu)
   - Charge state (e.g. C6+)
   - Energy range (MeV)

3. Select detector:
   - MCP (circular)
   - IP (rectangular)

4. Run simulation and inspect detector output

---

## Physics Model

Particle motion is governed by the Lorentz force:

```
F = q(E + v × B)
```

Regions:
- Magnetic field only
- Electric field only
- Field-free drift

Deflections:
- Magnetic ∝ q / (m v)
- Electric ∝ q / (m v²)

This produces the characteristic **parabolic separation** of ion species.

---

## Configuration Files

Simulation parameters are stored in JSON format:

```json
{
  "B": 0.63,
  "V_plate": 20000,
  "L_B": 50,
  "L_E": 150,
  "D_D": 740,
  "detector_type": "MCP"
}
```

---

## Use Cases

- Laser-driven ion acceleration experiments
- Thomson parabola calibration
- Multi-species beam analysis
- Teaching TP physics

---

## Author
Adrian McCay \
Queen's University Belfast \
Centre for Light-Matter Interactions

Developed for simulation and analysis of Thomson Parabola diagnostics in high-intensity laser experiments.
