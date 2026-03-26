import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit,
                             QComboBox, QPushButton, QGridLayout, QVBoxLayout, QFileDialog, QMessageBox,
                             QTableWidget, QTableWidgetItem, QSpinBox, QHeaderView)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import json
import os

# Constants
e = 1.602e-19

# Default Params
params = {
    'B': 0.6,
    'V_plate': 10000,
    'detector_offset_x': 20,
    'detector_offset_z': -20,
    'L_P': 300,
    'L_B': 50,
    'B_E': 25,
    'L_E': 100,
    'D_D': 500,
    'Ep_min': 10,
    'Ep_max': 80,
    'Ec_min': 10,
    'Ec_max': 30,
    'detector_radius': 30,
    'detector_width': 60,
    'detector_height': 60,
    'detector_type': 'MCP',
}

ions = {}

NUM = 20
DT = 1e-11
steps = 7000

# --- Placeholders for global variables ---
all_trajectories = []
detector_hits = {}
Y_det = 0

# === Simulation ===

def simulate_ion(name, mass, charge, energy_range_MeV):
    """
    Simulates ion trajectories through the Thomson Parabola spectrometer.
    
    PHYSICS OVERVIEW:
    The Thomson Parabola is a velocity selector and spectrometer that separates ions
    based on their mass-to-charge ratio. Ions experience three stages:
    1. Magnetic field region: Lorentz force causes deflection perpendicular to velocity
    2. Electric field region: Constant electric force perpendicular to beam direction
    3. Field-free drift region: Straight-line motion to detector (no forces)
    
    The parabolic pattern on the detector arises from the combined deflections in
    the magnetic and electric field regions followed by drift-space magnification.
    """
    energies = np.linspace(energy_range_MeV[0], energy_range_MeV[1], NUM)
    trajectories = []

    # Calculate region boundaries along Y (beam) direction
    Y_B_end = L_P + L_B                          # End of magnetic field region
    Y_E_start = Y_B_end + B_E                    # Start of electric field region
    Y_E_end = Y_E_start + L_E                    # End of electric field region

    for E_MeV in energies:
        # ============================================================
        # INITIAL CONDITIONS
        # ============================================================
        
        # Convert energy from MeV to Joules
        # E[J] = E[MeV] × 10^6 × e, where e is the elementary charge
        E_J = E_MeV * 1e6 * e
        
        # Calculate initial velocity using kinetic energy relation:
        # KE = (1/2)mv²
        # Solving for v: v = sqrt(2*KE/m) = sqrt(2*E/m)
        # This gives the speed at which the ion enters the magnetic field
        v0 = np.sqrt(2 * E_J / mass)
        
        # Initial position: at pinhole (origin of coordinate system)
        r = np.array([0.0, 0.0, 0.0])  # [x, y, z] in mm
        
        # Initial velocity: all along Y-axis (beam direction)
        # Ions start moving purely along the beam axis
        v = np.array([0.0, v0, 0.0])  # [vx, vy, vz] in m/s
        path = []

        # ============================================================
        # NUMERICAL INTEGRATION (Euler method)
        # ============================================================
        for _ in range(steps):
            y = r[1]  # Current position along beam axis
            
            # ========================================================
            # STAGE 1: MAGNETIC FIELD REGION
            # ========================================================
            # Region: L_P ≤ y < L_P + L_B
            # 
            # LORENTZ FORCE equation: F = q(v × B)
            # This is the fundamental force on a charged particle in a magnetic field.
            # 
            # Geometry in this code:
            # - Magnetic field B points in X direction: B = [B, 0, 0]
            # - Ions enter with velocity primarily in Y direction: v ≈ [0, v_y, v_z]
            # - Lorentz force F = q(v × B) points in Z direction initially
            # 
            # Physics result:
            # The Lorentz force is always perpendicular to velocity, causing the ion
            # to follow a circular arc in the Y-Z plane (deflection in Z direction
            # while maintaining beam velocity Y). The radius of curvature is:
            # r_c = m*v / (q*B)
            # 
            # Detector signature:
            # After exiting the magnetic field, the ion has acquired a Z-component of
            # velocity (v_z) proportional to B and time in field. In the drift region,
            # this transverse velocity causes the Z-deflection seen at the detector.
            # For a stronger B field, larger Z-deflection occurs (ion curves more).
            
            if L_P <= y < L_P + L_B:
                # F = q(v × B) where B = [B, 0, 0]
                # Cross product: v × B = [v_y*B_z - v_z*B_y, v_z*B_x - v_x*B_z, v_x*B_y - v_y*B_x]
                #                      = [0, -v_z*B, v_y*B]
                # Therefore: F = q[0, -v_z*B, v_y*B]
                # This gives acceleration in Y-Z plane, deflecting ion in Z direction
                F = charge * np.cross(v, np.array([B, 0, 0]))
            
            # ========================================================
            # STAGE 2: ELECTRIC FIELD REGION
            # ========================================================
            # Region: L_P + L_B + B_E ≤ y < L_P + L_B + B_E + L_E
            # 
            # ELECTRIC FORCE equation: F = qE
            # This is the force on a charged particle in an electric field.
            # 
            # Geometry in this code:
            # - Two parallel plates separated by 10 mm along X direction
            # - Electric field magnitude: E = V_plate / plate_separation
            # - Electric field points in X direction (perpendicular to beam)
            # - The field deflects ions in X direction based on their charge
            # 
            # Physics result:
            # Ions experience constant acceleration in X direction:
            # a_x = F_x / m = (q*E) / m = (q*V_plate/d) / m
            # This causes X-deflection to increase quadratically with time in field:
            # x(t) = (1/2)*a_x*t² = (q*V_plate)/(2*m*d) * t²
            # 
            # Detector signature:
            # The X-deflection depends on charge-to-mass ratio (q/m):
            # - Higher q/m → larger X-deflection
            # - Same energy but different q/m → separation in X direction
            # In the drift region, this transverse velocity v_x continues to increase
            # the X-position at the detector.
            # 
            # Combined effect (Parabola):
            # Z-deflection comes from magnetic field (proportional to B and v_y)
            # X-deflection comes from electric field (proportional to E and time)
            # The relationship between Z and X becomes parabolic because:
            # - Time in magnetic field determines Z-velocity
            # - Time in electric field determines X-velocity (and acceleration in X)
            # Both depend on ion energy E, creating the characteristic parabola
            
            elif L_P + L_B + B_E <= y < L_P + L_B + B_E + L_E:
                # Electric field magnitude: E = V_plate / d
                # where d = 10 mm = 10e-3 m (plate separation)
                E_field = V_plate / (10e-3)
                
                # Force in X direction only (perpendicular to beam)
                # F = q*E = q*V_plate/d pointing in X direction
                F = charge * np.array([E_field, 0, 0])
            
            # ========================================================
            # STAGE 3: FIELD-FREE DRIFT REGION
            # ========================================================
            # Region: y ≥ L_P + L_B + B_E + L_E
            # 
            # No electromagnetic forces act on the ion.
            # The ion travels in a straight line at constant velocity.
            # 
            # Physics result:
            # With no net force, the ion continues with the velocity it had when
            # exiting the electric field region. During the drift:
            # - Y-position increases at constant v_y (beam velocity)
            # - X-position increases at constant v_x (acquired in E-field)
            # - Z-position increases at constant v_z (acquired in B-field)
            # 
            # Detector signature (Magnification):
            # The drift distance D_D acts as a "drift tube" that magnifies the
            # transverse deflections (X and Z). An ion that acquired transverse
            # velocities v_x and v_z will have additional deflections during drift:
            # Δx_drift = v_x * (D_D / v_y)
            # Δz_drift = v_z * (D_D / v_y)
            # 
            # This magnification is what produces the widely spaced parabolic pattern
            # on the detector. Without the drift region, deflections would be smaller.
            
            else:
                F = np.zeros(3)

            # ============================================================
            # EQUATIONS OF MOTION (Euler Integration Method)
            # ============================================================
            
            # Newton's 2nd law: F = ma → a = F/m
            # Acceleration from the net force
            a = F / mass
            
            # Velocity update using Euler method:
            # v(t+dt) = v(t) + a*dt
            # This discretizes the differential equation dv/dt = a
            v += a * DT
            
            # Position update using Euler method:
            # r(t+dt) = r(t) + v*dt
            # This discretizes the differential equation dr/dt = v
            # 
            # Note: Factor of 1e3 converts velocity from m/s to mm/s
            # because all distances in this simulation are in millimeters
            r += v * DT * 1e3
            
            # Record position for trajectory visualization
            path.append(r.copy())
            
            # Stop condition 1: Ion reaches detector plane
            # When Y-position equals detector distance, trajectory is complete
            if r[1] >= L_P + L_B + B_E + L_E + D_D:
                break

            # Stop condition 2: Ion hits the E-plate electrodes
            # Collision detection: if ion position exceeds plate boundaries,
            # it is absorbed and trajectory terminates
            if Y_E_start <= r[1] <= Y_E_end:
                # E-plates: ±3 mm half-width in X, ±25 mm half-height in Z
                if abs(r[0]) > 3 or abs(r[2]) > 25:
                    break

        trajectories.append(np.array(path))

    return trajectories

def run_simulation():
    global all_trajectories, detector_hits, Y_det

    Y_B_end = L_P + L_B
    Y_E_start = Y_B_end + B_E
    Y_E_end = Y_E_start + L_E
    Y_det = Y_E_end + D_D

    all_trajectories = []
    detector_hits = {}

    for name, props in ions.items():
        trajs = simulate_ion(name, props['mass'], props['charge'], props['energy_range_MeV'])
        all_trajectories.append(trajs)
        detector_hits[name] = {
            'x': [path[-1][0] for path in trajs],
            'z': [path[-1][2] for path in trajs],
            'color': props['color']
        }

# === Plotting ===

def plot_3D_trajectories(save_video=False, video_filename='thomson_animation.mp4', dpi=150):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    Y_B_start = L_P
    Y_B_end = L_P + L_B
    Y_E_start = Y_B_end + B_E
    Y_E_end = Y_E_start + L_E
    Y_det = Y_E_end + D_D

    # === Magnetic Field Plates (gray slabs) ===
    for z_sign in [-1, 1]:
        Y, Z = np.meshgrid([Y_B_start, Y_B_end], [-25, 25])
        X = np.full_like(Y, z_sign * 4) #change the e-plate separation here
        ax.plot_surface(X, Y, Z, color='gray', alpha=0.6)

    # === Electric Field Plates (orange slabs) ===
    for z_sign in [-1, 1]:
        Y, Z = np.meshgrid([Y_E_start, Y_E_end], [-25, 25])
        X = np.full_like(Y, z_sign * 3)
        ax.plot_surface(X, Y, Z, color='orange', alpha=0.5)

    # === Detector outline ===
    theta = np.linspace(0, 2 * np.pi, 100)
    if detector_type == "MCP":
        ax.plot(detector_radius * np.cos(theta) + detector_offset_x,
                np.full_like(theta, Y_det),
                detector_radius * np.sin(theta) + detector_offset_z, 'b')
    elif detector_type == "IP":
        width, height = detector_width, detector_height
        corners = [[-width/2, -height/2], [width/2, -height/2],
                   [width/2, height/2], [-width/2, height/2], [-width/2, -height/2]]
        for i in range(4):
            x1 = corners[i][0] + detector_offset_x
            z1 = corners[i][1] + detector_offset_z
            x2 = corners[i+1][0] + detector_offset_x
            z2 = corners[i+1][1] + detector_offset_z
            ax.plot([x1, x2], [Y_det, Y_det], [z1, z2], 'b')

    # === Zero Point ===
    ax.scatter(0, Y_det, 0, color='black', marker='x', s=80, label="Zero Point")

    # === Trajectories ===
    lines = []
    for ion_props, trajs in zip(ions.values(), all_trajectories):
        for path in trajs:
            line, = ax.plot([], [], [], color=ion_props['color'], alpha=0.7)
            lines.append(line)

    def animate(frame):
        idx = 0
        for traj_group in all_trajectories:
            for path in traj_group:
                if frame < len(path):
                    segment = path[:frame * 15 + 1]
                else:
                    segment = path
                xs, ys, zs = zip(*segment)
                lines[idx].set_data(xs, ys)
                lines[idx].set_3d_properties(zs)
                idx += 1
        return lines
    
    if detector_type == "MCP":
        xlim = 1.5 * detector_radius
        zlim = 1.5 * detector_radius
    else:  # IP
        xlim = 1.5 * detector_width / 2
        zlim = 1.5 * detector_height / 2
    ax.set_xlim(detector_offset_x - xlim, detector_offset_x + xlim)
    ax.set_zlim(detector_offset_z - zlim, detector_offset_z + zlim)
    ax.set_ylim(0, Y_det)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.set_title('Thomson Parabola 3D Trajectories')
    ax.legend()

    # === SAVE VIDEO ===
    if save_video:
        print(f"Saving animation to {video_filename}...")
        # Using FFmpeg writer (ensure ffmpeg is installed on your system)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Thomson Parabola'), bitrate=5000)
        ani = animation.FuncAnimation(fig, animate, frames=5000, interval=5, blit=True, repeat=False)
        ani.save(video_filename, writer=writer, dpi=dpi)
        print(f"Video saved successfully to {video_filename}")
        plt.close(fig)
    else:
        ani = animation.FuncAnimation(fig, animate, frames=5000, interval=5, blit=True, repeat=False)
        plt.tight_layout()
        plt.show(block=True)
        plt.pause(5)
    
    plot_detector()

def plot_detector():
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.set_title(f"Detector Plane (Y = {Y_det:.0f} mm)")
    ax2.set_xlabel("X [mm]")
    ax2.set_ylabel("Z [mm]")
    ax2.set_aspect('equal')
    ax2.grid(True)

    # === Draw Detector Outline ===
    if detector_type == "MCP":
        detector_circle = plt.Circle((detector_offset_x, detector_offset_z),
                                     detector_radius, edgecolor='blue',
                                     facecolor='none', linewidth=1.5, label='Detector Edge')
        ax2.add_patch(detector_circle)
    elif detector_type == "IP":
        width, height = detector_width, detector_height  # Use global variables from GUI
        detector_rect = plt.Rectangle((detector_offset_x - width/2, detector_offset_z - height/2),
                                      width, height, edgecolor='blue',
                                      facecolor='none', linewidth=1.5, label='Detector Edge')
        ax2.add_patch(detector_rect)

    # === Zero point marker ===
    ax2.scatter(0, 0, color='black', marker='x', s=80, label='Zero Point')

    # === Hover annotation setup ===
    annot = ax2.annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                         bbox=dict(boxstyle="round", fc="w"),
                         arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    scatter_artists = []
    energy_lookup = []

        # === Plot hits and fits ===
    for name, data in detector_hits.items():
        x = np.array(data['x'])
        z = np.array(data['z'])
        color = data['color']
        
        # Get the number of nucleons (A) from the ions dictionary
        A = ions[name].get('A', 1)  # Default to 1 if not found
        
        # Calculate energies per nucleon from the stored energy range
        # The energy_range_MeV is stored as total energy, so divide by A to get per-nucleon
        E_min = ions[name].get('E_min', 10)
        E_max = ions[name].get('E_max', 30)
        energies = np.linspace(E_min, E_max, NUM)  # Energy per nucleon

        # Hit points
        scatter = ax2.scatter(x, z, color=color, label=f"{name} hits", picker=5)
        scatter_artists.append(scatter)
        energy_lookup.append(list(zip(x, z, energies)))

        # Parabola fit
        if len(x) >= 3:
            coeffs = np.polyfit(x, z, 2)
            x_fit = np.linspace(min(x), max(x), 200)
            z_fit = np.polyval(coeffs, x_fit)
            ax2.plot(x_fit, z_fit, '--', color=color, label=f"{name} fit")
            print(f"{name.upper()} parabola fit: z = {coeffs[0]:.3e} x² + {coeffs[1]:.3e} x + {coeffs[2]:.3e}")

    def update_annot(ind, scatter, energies):
        pos = scatter.get_offsets()[ind["ind"][0]]
        energy = energies[ind["ind"][0]][2]
        annot.xy = pos
        text = f"E = {energy:.1f} MeV/n"
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor('lightyellow')
        annot.set_alpha(0.9)

    def hover(event):
        visible = annot.get_visible()
        for scatter, energies in zip(scatter_artists, energy_lookup):
            cont, ind = scatter.contains(event)
            if cont:
                update_annot(ind, scatter, energies)
                annot.set_visible(True)
                fig2.canvas.draw_idle()
                return
        if visible:
            annot.set_visible(False)
            fig2.canvas.draw_idle()

    fig2.canvas.mpl_connect("motion_notify_event", hover)
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


class TPGui(QWidget):
    def __init__(self):
        super().__init__()
        self.entries = {}
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        grid = QGridLayout()
        label_map = {
            'B': "Magnetic Field B (T)",
            'V_plate': "Plate Voltage (V)",
            'detector_offset_x': "Detector X Offset (mm)",
            'detector_offset_z': "Detector Z Offset (mm)",
            'L_P': "Pinhole to Magnet (mm)",
            'L_B': "Magnet Length (mm)",
            'B_E': "Gap (Mag → E-Plates) (mm)",
            'L_E': "E-Plate Length (mm)",
            'D_D': "E-Plate to Detector (mm)",
        }

        for row, key in enumerate(label_map):
            grid.addWidget(QLabel(label_map[key]), row, 0)
            entry = QLineEdit(str(params[key]))
            grid.addWidget(entry, row, 1)
            self.entries[key] = entry

        # Detector type selection
        row += 1
        grid.addWidget(QLabel("Detector Type"), row, 0)
        self.detector_type_box = QComboBox()
        self.detector_type_box.addItems(["MCP", "IP"])
        self.detector_type_box.setCurrentText(params['detector_type'])
        grid.addWidget(self.detector_type_box, row, 1)

        # Radius / Width / Height
        row += 1
        self.radius_entry = QLineEdit(str(params['detector_radius']))
        self.width_entry = QLineEdit(str(params['detector_width']))
        self.height_entry = QLineEdit(str(params['detector_height']))

        self.radius_label = QLabel("Detector Radius")
        self.width_label = QLabel("IP Width (mm)")
        self.height_label = QLabel("IP Height (mm)")

        grid.addWidget(self.radius_label, row, 0)
        grid.addWidget(self.radius_entry, row, 1)
        grid.addWidget(self.width_label, row+1, 0)
        grid.addWidget(self.width_entry, row+1, 1)
        grid.addWidget(self.height_label, row+2, 0)
        grid.addWidget(self.height_entry, row+2, 1)

        self.detector_type_box.currentTextChanged.connect(self.toggle_detector_inputs)
        self.toggle_detector_inputs()

        layout.addLayout(grid)

        # === ION CONFIGURATION SECTION ===
        ion_label = QLabel("<b>Ion Configuration</b>")
        layout.addWidget(ion_label)

        # Create ion table
        self.ion_table = QTableWidget()
        self.ion_table.setColumnCount(6)
        self.ion_table.setHorizontalHeaderLabels(
            ["Ion Name", "A (Mass Number)", "Z (Charge)", "Color", "E_min (MeV/u)", "E_max (MeV/u)"]
        )
        self.ion_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ion_table.setMaximumHeight(200)
        layout.addWidget(self.ion_table)

        # Add ion button
        button_layout = QVBoxLayout()
        self.add_ion_button = QPushButton("Add Ion")
        self.add_ion_button.clicked.connect(self.add_ion_row)
        button_layout.addWidget(self.add_ion_button)

        self.remove_ion_button = QPushButton("Remove Selected Ion")
        self.remove_ion_button.clicked.connect(self.remove_ion_row)
        button_layout.addWidget(self.remove_ion_button)

        layout.addLayout(button_layout)

        # Add default ions
        self.add_default_ions()

        row = 0
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_sim)
        button_layout2 = QVBoxLayout()
        button_layout2.addWidget(self.run_button)

        # Save Video Button
        self.save_video_button = QPushButton("Save Animation as Video")
        self.save_video_button.clicked.connect(self.save_animation_video)
        button_layout2.addWidget(self.save_video_button)

        # Save and Load Settings buttons
        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self.save_settings)
        button_layout2.addWidget(self.save_button)

        self.load_button = QPushButton("Load Settings")
        self.load_button.clicked.connect(self.load_settings)
        button_layout2.addWidget(self.load_button)

        layout.addLayout(button_layout2)
        self.setLayout(layout)
        name_layout = QGridLayout()
        name_layout.addWidget(QLabel("A.McCay"), 0, 0)
        layout.addLayout(name_layout)

    def toggle_detector_inputs(self):
        dtype = self.detector_type_box.currentText()
        self.radius_label.setVisible(dtype == "MCP")
        self.radius_entry.setVisible(dtype == "MCP")
        self.width_label.setVisible(dtype == "IP")
        self.width_entry.setVisible(dtype == "IP")
        self.height_label.setVisible(dtype == "IP")
        self.height_entry.setVisible(dtype == "IP")

    def add_default_ions(self):
        """Add default ions (Proton and C6+) to the table"""
        # Proton
        self.add_ion_to_table("Proton", 1, 1, "red", 23, 80)
        # C6+
        self.add_ion_to_table("C6+", 12, 6, "blue", 10, 30)

    def add_ion_row(self):
        """Add a new empty ion row to the table"""
        self.add_ion_to_table("New Ion", 1, 1, "red", 10, 30)

    def add_ion_to_table(self, name, A, Z, color, E_min, E_max):
        """Add an ion row to the table with specified parameters"""
        row = self.ion_table.rowCount()
        self.ion_table.insertRow(row)

        # Ion name
        name_item = QTableWidgetItem(name)
        self.ion_table.setItem(row, 0, name_item)

        # Mass number (A)
        A_item = QTableWidgetItem(str(A))
        self.ion_table.setItem(row, 1, A_item)

        # Charge (Z)
        Z_item = QTableWidgetItem(str(Z))
        self.ion_table.setItem(row, 2, Z_item)

        # Color dropdown
        color_combo = QComboBox()
        colors = ["red", "blue", "green", "cyan", "magenta", "yellow", "orange", "purple", "brown", "pink"]
        color_combo.addItems(colors)
        color_combo.setCurrentText(color)
        self.ion_table.setCellWidget(row, 3, color_combo)

        # E_min
        E_min_item = QTableWidgetItem(str(E_min))
        self.ion_table.setItem(row, 4, E_min_item)

        # E_max
        E_max_item = QTableWidgetItem(str(E_max))
        self.ion_table.setItem(row, 5, E_max_item)

    def remove_ion_row(self):
        """Remove the selected ion row from the table"""
        current_row = self.ion_table.currentRow()
        if current_row >= 0:
            self.ion_table.removeRow(current_row)
        else:
            QMessageBox.warning(self, "Warning", "Please select an ion row to remove.")

    def get_ions_from_table(self):
        """Extract ion configuration from the table and populate the ions dictionary"""
        global ions
        ions = {}

        for row in range(self.ion_table.rowCount()):
            # Get values from table
            name = self.ion_table.item(row, 0).text()
            A = int(self.ion_table.item(row, 1).text())
            Z = int(self.ion_table.item(row, 2).text())
            color = self.ion_table.cellWidget(row, 3).currentText()
            E_min = float(self.ion_table.item(row, 4).text())
            E_max = float(self.ion_table.item(row, 5).text())

            # Calculate mass (in kg) from mass number A
            # A is number of nucleons (protons + neutrons)
            # Approximate atomic mass unit: 1 u = 1.66054e-27 kg
            mass = A * 1.66054e-27

            # Store ion configuration
            ions[name] = {
                'mass': mass,
                'charge': Z * e,
                'color': color,
                'energy_range_MeV': (E_min * A, E_max * A),  # Convert MeV/u to MeV
                'A': A,
                'Z': Z,
                'E_min': E_min,
                'E_max': E_max
            }

        return ions

    def run_sim(self):
        global B, V_plate, detector_offset_x, detector_offset_z
        global L_P, L_B, B_E, L_E, D_D, detector_radius, detector_type, detector_width, detector_height

        try:
            B = float(self.entries['B'].text())
            V_plate = float(self.entries['V_plate'].text())
            detector_offset_x = float(self.entries['detector_offset_x'].text())
            detector_offset_z = float(self.entries['detector_offset_z'].text())
            L_P = float(self.entries['L_P'].text())
            L_B = float(self.entries['L_B'].text())
            B_E = float(self.entries['B_E'].text())
            L_E = float(self.entries['L_E'].text())
            D_D = float(self.entries['D_D'].text())

            detector_type = self.detector_type_box.currentText()
            if detector_type == "MCP":
                detector_radius = float(self.radius_entry.text())
            else:
                detector_width = float(self.width_entry.text())
                detector_height = float(self.height_entry.text())

            # Get ions from table
            self.get_ions_from_table()
            
            if not ions:
                QMessageBox.warning(self, "Warning", "Please add at least one ion.")
                return

            run_simulation()
            plot_3D_trajectories(save_video=False)
        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Invalid input: {str(e)}")

    def save_animation_video(self):
        """Save the animation as a video file"""
        global B, V_plate, detector_offset_x, detector_offset_z
        global L_P, L_B, B_E, L_E, D_D, detector_radius, detector_type, detector_width, detector_height

        try:
            # First, update global variables from GUI
            B = float(self.entries['B'].text())
            V_plate = float(self.entries['V_plate'].text())
            detector_offset_x = float(self.entries['detector_offset_x'].text())
            detector_offset_z = float(self.entries['detector_offset_z'].text())
            L_P = float(self.entries['L_P'].text())
            L_B = float(self.entries['L_B'].text())
            B_E = float(self.entries['B_E'].text())
            L_E = float(self.entries['L_E'].text())
            D_D = float(self.entries['D_D'].text())

            detector_type = self.detector_type_box.currentText()
            if detector_type == "MCP":
                detector_radius = float(self.radius_entry.text())
            else:
                detector_width = float(self.width_entry.text())
                detector_height = float(self.height_entry.text())

            # Get ions from table
            self.get_ions_from_table()
            
            if not ions:
                QMessageBox.warning(self, "Warning", "Please add at least one ion.")
                return

            # Get filename from user
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Animation as Video",
                "thomson_animation.mp4",
                "MP4 Files (*.mp4);;AVI Files (*.avi);;GIF Files (*.gif)",
                options=options
            )

            if filename:
                # Run simulation
                run_simulation()
                # Save animation as video with high resolution (dpi=150 for better quality)
                plot_3D_trajectories(save_video=True, video_filename=filename, dpi=150)
                QMessageBox.information(self, "Success", f"Animation saved to:\n{filename}")
        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Invalid input: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save video:\n{str(e)}\n\nNote: FFmpeg must be installed on your system.")

    def save_settings(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self, "Save Settings", "settings/", "JSON Files (*.json)", options=options)
        if filename:
            try:
                settings = {key: float(self.entries[key].text()) for key in self.entries}
                settings['detector_type'] = self.detector_type_box.currentText()
                if settings['detector_type'] == 'MCP':
                    settings['detector_radius'] = float(self.radius_entry.text())
                else:
                    settings['detector_width'] = float(self.width_entry.text())
                    settings['detector_height'] = float(self.height_entry.text())

                # Save ion configuration
                ions_config = []
                for row in range(self.ion_table.rowCount()):
                    ion_data = {
                        'name': self.ion_table.item(row, 0).text(),
                        'A': int(self.ion_table.item(row, 1).text()),
                        'Z': int(self.ion_table.item(row, 2).text()),
                        'color': self.ion_table.cellWidget(row, 3).currentText(),
                        'E_min': float(self.ion_table.item(row, 4).text()),
                        'E_max': float(self.ion_table.item(row, 5).text()),
                    }
                    ions_config.append(ion_data)
                settings['ions'] = ions_config

                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w') as f:
                    json.dump(settings, f, indent=4)
                QMessageBox.information(self, "Success", "Settings saved successfully.")
            except ValueError as e:
                QMessageBox.critical(self, "Error", f"Invalid input: {str(e)}")

    def load_settings(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Load Settings", "settings/", "JSON Files (*.json)", options=options)
        if filename:
            try:
                with open(filename, 'r') as f:
                    settings = json.load(f)
                for key in self.entries:
                    self.entries[key].setText(str(settings[key]))
                self.detector_type_box.setCurrentText(settings['detector_type'])
                if settings['detector_type'] == 'MCP':
                    self.radius_entry.setText(str(settings['detector_radius']))
                else:
                    self.width_entry.setText(str(settings['detector_width']))
                    self.height_entry.setText(str(settings['detector_height']))

                # Load ion configuration
                self.ion_table.setRowCount(0)  # Clear existing rows
                if 'ions' in settings:
                    for ion_data in settings['ions']:
                        self.add_ion_to_table(
                            ion_data['name'],
                            ion_data['A'],
                            ion_data['Z'],
                            ion_data['color'],
                            ion_data['E_min'],
                            ion_data['E_max']
                        )

                self.toggle_detector_inputs()
                QMessageBox.information(self, "Loaded", "Settings loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load settings: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = TPGui()
    gui.show()
    sys.exit(app.exec_())