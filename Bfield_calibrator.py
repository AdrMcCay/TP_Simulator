#!/usr/bin/env python3
"""
TP calibration helper
- Loads TP geometry from JSON (user-specified path).
- Toggle the main-ion to calibrate via ION_TO_CALIBRATE ('C6+' or 'proton').
- When run, opens a small GUI: paste lines "E, deflection" (one per line).
  * For C6+: enter Energy in MeV/u (e.g. 12.0), Deflection in mm (e.g. 15.3)
  * For proton: enter Energy in MeV (total), Deflection in mm
- For each pair the script fits a B (T) such that simulated radial deflection at the detector
  matches the measured deflection. Prints per-point B and mean B.
- Shows a detector-plane plot of simulated traces and overlays measured points.
"""
import json
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

ION_TO_CALIBRATE = 'C6+'   # 'C6+' or 'proton'

#  constants
e = 1.602176634e-19
amu = 1.66053906660e-27
m_proton = 1.67262192369e-27

# Ion definitions
IONS = {
    'proton': {'mass': m_proton, 'charge': 1 * e, 'label': 'proton', 'energy_units': 'MeV'},
    'C6+': {'mass': 12 * amu, 'charge': 6 * e, 'label': 'C6+', 'energy_units': 'MeV/u', 'A': 12},
}

# Simulation stepping
NUM_TRACE_ENERGIES = 1000   # number of simulated energies to show traces
DT = 1e-11
STEPS = 120000  # simulation steps (tweak if you need faster/slower) 

def load_tp_params():
    """Load TP settings from JSON file (dialog or default path)"""
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Select TP JSON settings file",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    root.destroy()
    
    if not filepath:
        raise FileNotFoundError("No JSON file selected")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"TP JSON path not found: {filepath}")
    
    with open(filepath, 'r') as f:
        p = json.load(f)
    print(f"Loaded TP parameters from: {filepath}")
    return p

# ---------------- core: single-energy single-B simulation ----------------

def simulate_deflection_mm(mass, charge, energy_MeV_total, B_T, params):
    """
    Simulate single ion with total kinetic energy energy_MeV_total (MeV)
    mass in kg, charge in C, through the TP geometry stored in params (mm units).
    Returns radial deflection at detector in mm (sqrt(x^2 + z^2)).
    """
    # positions are tracked in mm, velocity in m/s
    r = np.array([0.0, 0.0, 0.0])  # x, y, z in mm (y is beam axis)
    E_J = energy_MeV_total * 1e6 * e
    v = np.array([0.0, np.sqrt(2 * E_J / mass), 0.0])  # m/s
    
    L_P = params['L_P']     # mm
    L_B = params['L_B']     # mm
    B_E = params['B_E']     # mm
    L_E = params['L_E']     # mm
    D_D = params['D_D']     # mm
    V_plate = params.get('V_plate', 0.0)
    
    Y_B_end = L_P + L_B
    Y_E_start = Y_B_end + B_E
    Y_E_end = Y_E_start + L_E
    Y_det = Y_E_end + D_D
    
    for _ in range(STEPS):
        y = r[1]
        if L_P <= y < L_P + L_B:
            #B along +X
            F = charge * np.cross(v, np.array([B_T, 0.0, 0.0]))
        elif Y_E_start <= y < Y_E_end:
            E_field = V_plate / (10e-3) if V_plate else 0.0
            F = charge * np.array([E_field, 0.0, 0.0])
        else:
            F = np.zeros(3)
        
        a = F / mass
        v += a * DT
        r += v * DT * 1e3
        
        # stop when reaching detector plane
        if r[1] >= Y_det:
            break
    
    return float(np.sqrt(r[0]**2 + r[2]**2))

#fitting
def fit_B_for_pair(ion_key, E_input, measured_defl_mm, params, B_bounds=(0.1, 1)):
    """
    For a single (E_input, deflection) pair, find B that reproduces deflection.
    E_input should be:
      - for 'C6+': MeV/u (we convert to total MeV by multiplying A)
      - for 'proton': total MeV
    Returns fitted_B (float, Tesla) and simulated deflection at that B.
    """
    ion = IONS[ion_key]
    if ion_key == 'C6+':
        energy_total = float(E_input) * ion['A']
    else:
        energy_total = float(E_input)
    
    # objective: absolute difference between simulated and measured deflection
    def objective(Btest):
        simd = simulate_deflection_mm(ion['mass'], ion['charge'], energy_total, Btest, params)
        return abs(simd - measured_defl_mm)
    
    res = minimize_scalar(objective, bounds=B_bounds, method='bounded', options={'xatol':1e-5})
    return float(res.x), simulate_deflection_mm(ion['mass'], ion['charge'], energy_total, res.x, params)

# GUI 
def open_input_gui_and_get_pairs(default_ion_key):
    result = {'pairs': None, 'ion_key': default_ion_key, 'params_file': None}
    root = tk.Tk()
    root.title("TP calibration input")
    root.geometry("520x420")
    
    # File selection section
    file_frame = tk.Frame(root)
    file_frame.pack(padx=10, pady=6)
    tk.Label(file_frame, text="TP Settings File:").pack(side=tk.LEFT)
    file_label = tk.Label(file_frame, text="(None selected)", fg="red")
    file_label.pack(side=tk.LEFT, padx=5)
    
    def select_file():
        filepath = filedialog.askopenfilename(
            title="Select TP JSON settings file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            result['params_file'] = filepath
            file_label.config(text=os.path.basename(filepath), fg="green")
    
    tk.Button(file_frame, text="Browse", command=select_file).pack(side=tk.LEFT, padx=5)
    
    label = tk.Label(root, text=(
        "Paste lines with: Energy, Deflection_mm\n"
        "For C6+: Energy in MeV/u (e.g. 12.0). For proton: MeV.\n"
        "Examples:\n  12.0, 14.2\n  15  16.0\n"))
    label.pack(padx=10, pady=6)
    
    # Ion selection (defaults to main toggle)
    ion_frame = tk.Frame(root)
    ion_frame.pack(padx=10, pady=4)
    tk.Label(ion_frame, text="Ion:").pack(side=tk.LEFT)
    ion_var = tk.StringVar(value=default_ion_key)
    tk.Radiobutton(ion_frame, text="C6+", variable=ion_var, value='C6+').pack(side=tk.LEFT)
    tk.Radiobutton(ion_frame, text="proton", variable=ion_var, value='proton').pack(side=tk.LEFT)
    
    text = tk.Text(root, height=12, width=70)
    text.pack(padx=10, pady=6)
    
    # paste a small template
    text.insert("1.0", "7.0, 38.3\n10.0, 32.0\n19.2, 23.2")
    
    def on_ok():
        if not result['params_file']:
            messagebox.showerror("Error", "Please select a TP settings file first!")
            return
        
        raw = text.get("1.0", tk.END).strip()
        ion_sel = ion_var.get()
        pairs = []
        if raw:
            for line in raw.splitlines():
                s = line.strip()
                if not s: 
                    continue
                s = s.replace(',', ' ')
                parts = s.split()
                if len(parts) < 2:
                    continue
                try:
                    E = float(parts[0])
                    D = float(parts[1])
                    pairs.append((ion_sel, E, D))
                except ValueError:
                    continue
        result['pairs'] = pairs
        result['ion_key'] = ion_sel
        root.destroy()
    
    def on_cancel():
        result['pairs'] = []
        root.destroy()
    
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=8)
    tk.Button(btn_frame, text="OK", width=12, command=on_ok).pack(side=tk.LEFT, padx=8)
    tk.Button(btn_frame, text="Cancel", width=12, command=on_cancel).pack(side=tk.LEFT, padx=8)
    
    root.mainloop()
    return result['pairs'], result['ion_key'], result['params_file']

# Plotting 
def plot_traces_and_points(params, ion_key, measured_points, B_T):
    """
    Simulates a trace for the chosen ion and B_T,
    """
    ion = IONS[ion_key]
    if ion_key == 'C6+':
        E_min = 5.0    # MeV/u
        E_max = 50.0    # MeV/u
        energy_list = np.linspace(E_min, E_max, NUM_TRACE_ENERGIES)
        # convert each to total MeV for simulation
        energy_list_total = [E * ion['A'] for E in energy_list]
    else:
        E_min = 23.0 # MeV
        E_max = 80.0 # MeV
        energy_list_total = np.linspace(E_min, E_max, NUM_TRACE_ENERGIES)
    
    sim_x = []
    sim_z = []
    for Etot in energy_list_total:
        r = np.array([0.0, 0.0, 0.0])
        E_J = Etot * 1e6 * e
        v = np.array([0.0, np.sqrt(2 * E_J / ion['mass']), 0.0])
        L_P = params['L_P']; L_B = params['L_B']; B_E = params['B_E']
        L_E = params['L_E']; D_D = params['D_D']; V_plate = params.get('V_plate', 0.0)
        Y_B_end = L_P + L_B; Y_E_start = Y_B_end + B_E; Y_E_end = Y_E_start + L_E
        Y_det = Y_E_end + D_D
        for _ in range(STEPS):
            y = r[1]
            if L_P <= y < L_P + L_B:
                F = ion['charge'] * np.cross(v, np.array([B_T, 0.0, 0.0]))
            elif Y_E_start <= y < Y_E_end:
                E_field = V_plate / (10e-3) if V_plate else 0.0
                F = ion['charge'] * np.array([E_field, 0.0, 0.0])
            else:
                F = np.zeros(3)
            a = F / ion['mass']
            v += a * DT
            r += v * DT * 1e3
            if r[1] >= Y_det:
                break
        sim_x.append(float(r[0])); sim_z.append(float(r[2]))
    
    # Plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')
    ax.grid(True)
    ax.scatter(sim_x, sim_z, c='tab:blue', s=18, label='Simulated trace (energy sweep)')
    ax.set_xlim(-5,5)
    
    meas_x = []
    meas_z = []
    for ion_sel, E, D in measured_points:
        ion = IONS[ion_sel]
        # Convert energy to total MeV
        if ion_sel == 'C6+':
            Etot = E * ion['A']
        else:
            Etot = E
        # Run full simulation with fitted B_T
        r = np.array([0.0, 0.0, 0.0])
        E_J = Etot * 1e6 * e
        v = np.array([0.0, np.sqrt(2 * E_J / ion['mass']), 0.0])
        L_P = params['L_P']; L_B = params['L_B']; B_E = params['B_E']
        L_E = params['L_E']; D_D = params['D_D']; V_plate = params.get('V_plate', 0.0)
        Y_B_end = L_P + L_B; Y_E_start = Y_B_end + B_E; Y_E_end = Y_E_start + L_E
        Y_det = Y_E_end + D_D
        for _ in range(STEPS):
            y = r[1]
            if L_P <= y < L_P + L_B:
                F = ion['charge'] * np.cross(v, np.array([B_T, 0.0, 0.0]))
            elif Y_E_start <= y < Y_E_end:
                E_field = V_plate / (10e-3) if V_plate else 0.0
                F = ion['charge'] * np.array([E_field, 0.0, 0.0])
            else:
                F = np.zeros(3)
            a = F / ion['mass']
            v += a * DT
            r += v * DT * 1e3
            if r[1] >= Y_det:
                break
        meas_x.append(float(r[0]))
        meas_z.append(float(r[2]))
    
    if meas_x:
        ax.scatter(meas_x, meas_z, c='red', s=60, marker='x', label='Measured points (radial)')
    ax.set_xlabel('X [mm]'); ax.set_ylabel('Z [mm]')
    ax.set_title(f"Detector-plane: simulated traces @ B={B_T:.4f} T (ion={ion_key})")
    ax.legend()
    plt.show()

def main():
    print(f"Main code ion toggle: {ION_TO_CALIBRATE}. You can change this variable in the script if you want a different default.")
    user_pairs, ion_selected, params_file = open_input_gui_and_get_pairs(ION_TO_CALIBRATE)
    
    if not user_pairs:
        print("No calibration pairs provided; exiting.")
        return
    
    if not params_file:
        print("No settings file selected; exiting.")
        return
    
    # Load params from selected file
    if not os.path.exists(params_file):
        print(f"Settings file not found: {params_file}")
        return
    
    with open(params_file, 'r') as f:
        params = json.load(f)
    print(f"Loaded TP parameters from: {params_file}")
    
    # normalize/ensure required keys exist
    # if some keys missing, provide defaults (keeps backwards compatibility)
    for k,v in [('L_P',370.0),('L_B',50.0),('B_E',25.0),('L_E',300.0),('D_D',570.0),
                    ('V_plate',0.0),('Ep_min',23.0),('Ep_max',150.0),('Ec_min',10.0),('Ec_max',30.0)]:
        params.setdefault(k, v)
    
    # Force V_plate to 0V regardless of settings file
    params['V_plate'] = 0.0
    
    # Fit B for each pair
    fitted_Bs = []
    results = []
    print("\nFitting B for provided pairs (this may take a moment)...")
    for (ion_key, E, D) in user_pairs:
        ion_key = ion_key if ion_key in IONS else ION_TO_CALIBRATE
        try:
            Bfit, simd = fit_B_for_pair(ion_key, E, D, params, B_bounds=(0.3, 0.8))
            print(f"  Ion={ion_key:4s}  E={E:7.3f} ({'MeV/u' if ion_key=='C6+' else 'MeV'})  measured defl={D:7.3f} mm  ->  B_fit={Bfit:8.5f} T  simulated_defl_at_B={simd:7.3f} mm")
            fitted_Bs.append(Bfit)
            results.append((ion_key, E, D, Bfit, simd))
        except Exception as exc:
            print("  Fit failed for entry:", ion_key, E, D, "Error:", exc)
    
    if not fitted_Bs:
        print("No successful fits. Exiting.")
        return
    
    meanB = float(np.mean(fitted_Bs))
    print(f"\nMean fitted B from {len(fitted_Bs)} points: {meanB:.6f} T")
    
    try:
        root = tk.Tk(); root.withdraw()
        messagebox.showinfo("Calibration result", f"Mean fitted B = {meanB:.6f} T")
        root.destroy()
    except Exception:
        pass
    
    params['B'] = meanB
    
    # Plot simulated traces and overlay measured points for visual comparison
    plot_traces_and_points(params, ion_selected if ion_selected in IONS else ION_TO_CALIBRATE, [(ik, e, d) for (ik,e,d) in user_pairs], meanB)

if __name__ == '__main__':
    main()