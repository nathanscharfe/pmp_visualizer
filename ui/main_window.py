import sys
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, 
    QGroupBox, QSlider, QComboBox, QCheckBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from core.solver import DoubleIntegratorProblem


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PMP Visualizer - Double Integrator")
        self.setGeometry(100, 100, 1600, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        
        # Right panel - Plots
        right_panel = self.create_plot_panel()
        
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 3)
        
        central_widget.setLayout(main_layout)
        
        # Initialize solutions
        self.optimal_solution = None
        self.manual_solution = None
    
    def create_control_panel(self):
        """Create the left control panel."""
        main_group = QGroupBox("Settings")
        layout = QVBoxLayout()
        
        # ===== Problem Statement =====
        problem_text = """
        <b>Double Integrator Optimal Control Problem</b><br><br>
        
        <b>System Dynamics:</b><br>
        ẋ₁ = x₂<br>
        ẋ₂ = u<br><br>
        
        <b>Control Constraints:</b><br>
        |u| ≤ u_max<br><br>
        
        <b>Cost Function:</b><br>
        J = ∫ u²(t) dt<br><br>
        
        <b>Boundary Conditions:</b><br>
        x(0) = [x₀, ẋ₀]<br>
        x(T) = [xf, ẋf]<br><br>
        
        <b>Goal:</b> Find control u(t) that minimizes energy cost<br>
        while satisfying boundary conditions.
        """
        
        problem_label = QLabel(problem_text)
        problem_label.setWordWrap(True)
        problem_label.setStyleSheet("border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9;")
        layout.addWidget(problem_label)
        
        layout.addWidget(QLabel("\n━━ Problem Parameters ━━"))
        
        # Initial position
        layout.addWidget(QLabel("Initial Position (x₀):"))
        self.x0_spin = QDoubleSpinBox()
        self.x0_spin.setRange(-10, 10)
        self.x0_spin.setValue(0.0)
        self.x0_spin.setSingleStep(0.1)
        layout.addWidget(self.x0_spin)
        
        # Initial velocity
        layout.addWidget(QLabel("Initial Velocity (ẋ₀):"))
        self.v0_spin = QDoubleSpinBox()
        self.v0_spin.setRange(-5, 5)
        self.v0_spin.setValue(0.0)
        self.v0_spin.setSingleStep(0.1)
        layout.addWidget(self.v0_spin)
        
        # Final position
        layout.addWidget(QLabel("Final Position (xf):"))
        self.xf_spin = QDoubleSpinBox()
        self.xf_spin.setRange(-10, 10)
        self.xf_spin.setValue(5.0)
        self.xf_spin.setSingleStep(0.1)
        layout.addWidget(self.xf_spin)
        
        # Final velocity
        layout.addWidget(QLabel("Final Velocity (ẋf):"))
        self.vf_spin = QDoubleSpinBox()
        self.vf_spin.setRange(-5, 5)
        self.vf_spin.setValue(0.0)
        self.vf_spin.setSingleStep(0.1)
        layout.addWidget(self.vf_spin)
        
        # Time horizon
        layout.addWidget(QLabel("Time Horizon (T):"))
        self.T_spin = QDoubleSpinBox()
        self.T_spin.setRange(0.1, 20)
        self.T_spin.setValue(5.0)
        self.T_spin.setSingleStep(0.1)
        layout.addWidget(self.T_spin)
        
        # Control bound
        layout.addWidget(QLabel("Control Bound (|u|):"))
        self.umax_spin = QDoubleSpinBox()
        self.umax_spin.setRange(0.1, 10)
        self.umax_spin.setValue(1.0)
        self.umax_spin.setSingleStep(0.1)
        layout.addWidget(self.umax_spin)
        
        # Solve button
        self.solve_button = QPushButton("Solve Optimal Problem")
        self.solve_button.clicked.connect(self.solve_optimal)
        layout.addWidget(self.solve_button)
        
        # ===== Manual Control =====
        layout.addWidget(QLabel("\n━━ Manual Control ━━"))
        
        self.manual_enabled = QCheckBox("Enable Manual Control")
        self.manual_enabled.setChecked(False)
        self.manual_enabled.stateChanged.connect(self.on_manual_changed)
        layout.addWidget(self.manual_enabled)
        
        layout.addWidget(QLabel("Control Type:"))
        self.manual_type = QComboBox()
        self.manual_type.addItems(["Constant", "Linear Ramp", "Sine Wave", "Step"])
        layout.addWidget(self.manual_type)
        
        layout.addWidget(QLabel("Control Amplitude:"))
        self.manual_amp = QDoubleSpinBox()
        self.manual_amp.setRange(-1.0, 1.0)
        self.manual_amp.setValue(0.5)
        self.manual_amp.setSingleStep(0.1)
        layout.addWidget(self.manual_amp)
        
        layout.addWidget(QLabel("Frequency (for sine):"))
        self.manual_freq = QDoubleSpinBox()
        self.manual_freq.setRange(0.1, 5.0)
        self.manual_freq.setValue(1.0)
        self.manual_freq.setSingleStep(0.1)
        layout.addWidget(self.manual_freq)
        
        self.update_manual_button = QPushButton("Update Manual Solution")
        self.update_manual_button.clicked.connect(self.update_manual)
        self.update_manual_button.setEnabled(False)
        layout.addWidget(self.update_manual_button)
        
        # ===== Cost Comparison =====
        layout.addWidget(QLabel("\n━━ Results ━━"))
        layout.addWidget(QLabel("Optimal Cost (Energy):"))
        self.optimal_cost_label = QLabel("—")
        layout.addWidget(self.optimal_cost_label)
        
        layout.addWidget(QLabel("Manual Cost (Energy):"))
        self.manual_cost_label = QLabel("—")
        layout.addWidget(self.manual_cost_label)
        
        layout.addWidget(QLabel("Cost Ratio (Manual/Optimal):"))
        self.ratio_label = QLabel("—")
        layout.addWidget(self.ratio_label)
        
        layout.addStretch()
        main_group.setLayout(layout)
        return main_group
    
    def create_plot_panel(self):
        """Create the right plot panel with 6 subplots."""
        # Create figure with 6 subplots (2x3)
        self.fig = Figure(figsize=(11, 8), dpi=100)
        self.ax1 = self.fig.add_subplot(2, 3, 1)  # State trajectory
        self.ax2 = self.fig.add_subplot(2, 3, 2)  # Control
        self.ax3 = self.fig.add_subplot(2, 3, 3)  # Phase portrait
        self.ax4 = self.fig.add_subplot(2, 3, 4)  # Costate (p)
        self.ax5 = self.fig.add_subplot(2, 3, 5)  # Hamiltonian
        self.ax6 = self.fig.add_subplot(2, 3, 6)  # Cost comparison
        
        self.canvas = FigureCanvas(self.fig)
        
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        widget.setLayout(layout)
        
        return widget
    
    def on_manual_changed(self):
        """Enable/disable manual control inputs."""
        enabled = self.manual_enabled.isChecked()
        self.manual_type.setEnabled(enabled)
        self.manual_amp.setEnabled(enabled)
        self.manual_freq.setEnabled(enabled)
        self.update_manual_button.setEnabled(enabled)
    
    def solve_optimal(self):
        """Solve the optimal control problem."""
        try:
            x0 = [self.x0_spin.value(), self.v0_spin.value()]
            xf = [self.xf_spin.value(), self.vf_spin.value()]
            T = self.T_spin.value()
            umax = self.umax_spin.value()
            
            problem = DoubleIntegratorProblem(x0, xf, T, umax)
            self.optimal_solution = problem.solve()
            
            # Compute cost for optimal
            u = self.optimal_solution['u']
            t = self.optimal_solution['t']
            optimal_cost = np.trapezoid(u**2, t)
            self.optimal_cost_label.setText(f"{optimal_cost:.4f}")
            
            self.update_plots()
        except Exception as e:
            print(f"Error solving optimal problem: {e}")
            import traceback
            traceback.print_exc()
    
    def update_manual(self):
        """Compute manual control solution."""
        if self.optimal_solution is None:
            print("Solve optimal problem first!")
            return
        
        try:
            T = self.T_spin.value()
            control_type = self.manual_type.currentText()
            amp = self.manual_amp.value()
            freq = self.manual_freq.value()
            
            # Define control function based on type
            if control_type == "Constant":
                u_func = lambda t: amp
            elif control_type == "Linear Ramp":
                u_func = lambda t: amp * (2 * t / T - 1)
            elif control_type == "Sine Wave":
                u_func = lambda t: amp * np.sin(2 * np.pi * freq * t / T)
            elif control_type == "Step":
                u_func = lambda t: amp if t < T/2 else -amp
            
            # Simulate with manual control
            x0 = [self.x0_spin.value(), self.v0_spin.value()]
            xf = [self.xf_spin.value(), self.vf_spin.value()]
            umax = self.umax_spin.value()
            
            problem = DoubleIntegratorProblem(x0, xf, T, umax)
            self.manual_solution = problem.simulate_with_control(u_func)
            
            # Update cost labels
            manual_cost = self.manual_solution['cost']
            optimal_cost = float(self.optimal_cost_label.text())
            ratio = manual_cost / optimal_cost if optimal_cost > 0 else 0
            
            self.manual_cost_label.setText(f"{manual_cost:.4f}")
            self.ratio_label.setText(f"{ratio:.2f}x")
            
            self.update_plots()
        except Exception as e:
            print(f"Error computing manual solution: {e}")
            import traceback
            traceback.print_exc()
    
    def update_plots(self):
        """Update all plots with solutions."""
        if self.optimal_solution is None:
            return
        
        # Clear all axes
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
            ax.clear()
        
        t_opt = self.optimal_solution['t']
        x_opt = self.optimal_solution['x']
        u_opt = self.optimal_solution['u']
        p_opt = self.optimal_solution['p']
        H_opt = self.optimal_solution['H']
        
        # ===== Plot 1: State Trajectory =====
        self.ax1.plot(t_opt, x_opt[:, 0], 'b-', label='Position (Optimal)', linewidth=2)
        self.ax1.plot(t_opt, x_opt[:, 1], 'r-', label='Velocity (Optimal)', linewidth=2)
        
        if self.manual_solution is not None:
            t_man = self.manual_solution['t']
            x_man = self.manual_solution['x']
            self.ax1.plot(t_man, x_man[:, 0], 'b--', label='Position (Manual)', linewidth=1.5, alpha=0.7)
            self.ax1.plot(t_man, x_man[:, 1], 'r--', label='Velocity (Manual)', linewidth=1.5, alpha=0.7)
        
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('State')
        self.ax1.set_title('State Trajectory')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # ===== Plot 2: Control =====
        self.ax2.plot(t_opt, u_opt, 'g-', linewidth=2.5, label='Optimal Control')
        
        # Plot manual control if available
        if self.manual_solution is not None:
            t_man = self.manual_solution['t']
            u_man = self.manual_solution['u']
            self.ax2.plot(t_man, u_man, 'orange', linewidth=1.5, linestyle='--', 
                         label='Manual Control', alpha=0.7)
        
        self.ax2.axhline(self.umax_spin.value(), color='k', linestyle='--', alpha=0.3)
        self.ax2.axhline(-self.umax_spin.value(), color='k', linestyle='--', alpha=0.3)
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Control u(t)')
        self.ax2.set_title('Control Input')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        # ===== Plot 3: Phase Portrait =====
        self.ax3.plot(x_opt[:, 0], x_opt[:, 1], 'b-', linewidth=2, label='Optimal')
        self.ax3.plot(x_opt[0, 0], x_opt[0, 1], 'go', markersize=10, label='Start')
        self.ax3.plot(x_opt[-1, 0], x_opt[-1, 1], 'ro', markersize=10, label='End')
        
        if self.manual_solution is not None:
            x_man = self.manual_solution['x']
            self.ax3.plot(x_man[:, 0], x_man[:, 1], 'orange', linewidth=1.5, 
                         linestyle='--', label='Manual', alpha=0.7)
        
        self.ax3.set_xlabel('Position')
        self.ax3.set_ylabel('Velocity')
        self.ax3.set_title('Phase Portrait')
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
        
        # ===== Plot 4: Costate (p) =====
        self.ax4.plot(t_opt, p_opt[:, 0], 'b-', label='p₁', linewidth=2)
        self.ax4.plot(t_opt, p_opt[:, 1], 'r-', label='p₂', linewidth=2)
        self.ax4.set_xlabel('Time (s)')
        self.ax4.set_ylabel('Costate')
        self.ax4.set_title('Costate Trajectory (p)\n(Optimal Solution Only)')
        self.ax4.legend()
        self.ax4.grid(True, alpha=0.3)
        
        # ===== Plot 5: Hamiltonian =====
        self.ax5.plot(t_opt, H_opt, 'purple', linewidth=2)
        self.ax5.set_xlabel('Time (s)')
        self.ax5.set_ylabel('H(t)')
        self.ax5.set_title('Hamiltonian\n(Optimal Solution Only)')
        self.ax5.grid(True, alpha=0.3)
        
        # ===== Plot 6: Cost Comparison =====
        if self.manual_solution is not None:
            costs = [
                float(self.optimal_cost_label.text()),
                float(self.manual_cost_label.text())
            ]
            labels = ['Optimal', 'Manual']
            colors = ['green', 'orange']
            bars = self.ax6.bar(labels, costs, color=colors, alpha=0.7, edgecolor='black')
            
            # Add value labels on bars
            for bar, cost in zip(bars, costs):
                height = bar.get_height()
                self.ax6.text(bar.get_x() + bar.get_width()/2., height,
                            f'{cost:.4f}',
                            ha='center', va='bottom')
            
            self.ax6.set_ylabel('Cost (Energy)')
            self.ax6.set_title('Cost Comparison')
            self.ax6.grid(True, alpha=0.3, axis='y')
        else:
            self.ax6.text(0.5, 0.5, 'Enable manual control\nto compare costs',
                         ha='center', va='center', transform=self.ax6.transAxes,
                         fontsize=11, style='italic', color='gray')
            self.ax6.set_title('Cost Comparison')
        
        self.fig.tight_layout()
        self.canvas.draw()


def main():
    app = __import__('PyQt5.QtWidgets', fromlist=['QApplication']).QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
    
    def create_control_panel(self):
        """Create the left control panel."""
        group = QGroupBox("Problem Parameters")
        layout = QVBoxLayout()
        
        # Initial position
        layout.addWidget(QLabel("Initial Position (x₀):"))
        self.x0_spin = QDoubleSpinBox()
        self.x0_spin.setRange(-10, 10)
        self.x0_spin.setValue(0.0)
        self.x0_spin.setSingleStep(0.1)
        layout.addWidget(self.x0_spin)
        
        # Initial velocity
        layout.addWidget(QLabel("Initial Velocity (ẋ₀):"))
        self.v0_spin = QDoubleSpinBox()
        self.v0_spin.setRange(-5, 5)
        self.v0_spin.setValue(0.0)
        self.v0_spin.setSingleStep(0.1)
        layout.addWidget(self.v0_spin)
        
        # Final position
        layout.addWidget(QLabel("Final Position (xf):"))
        self.xf_spin = QDoubleSpinBox()
        self.xf_spin.setRange(-10, 10)
        self.xf_spin.setValue(5.0)
        self.xf_spin.setSingleStep(0.1)
        layout.addWidget(self.xf_spin)
        
        # Final velocity
        layout.addWidget(QLabel("Final Velocity (ẋf):"))
        self.vf_spin = QDoubleSpinBox()
        self.vf_spin.setRange(-5, 5)
        self.vf_spin.setValue(0.0)
        self.vf_spin.setSingleStep(0.1)
        layout.addWidget(self.vf_spin)
        
        # Time horizon
        layout.addWidget(QLabel("Time Horizon (T):"))
        self.T_spin = QDoubleSpinBox()
        self.T_spin.setRange(0.1, 20)
        self.T_spin.setValue(5.0)
        self.T_spin.setSingleStep(0.1)
        layout.addWidget(self.T_spin)
        
        # Control bound
        layout.addWidget(QLabel("Control Bound (|u|):"))
        self.umax_spin = QDoubleSpinBox()
        self.umax_spin.setRange(0.1, 10)
        self.umax_spin.setValue(1.0)
        self.umax_spin.setSingleStep(0.1)
        layout.addWidget(self.umax_spin)
        
        # Solve button
        self.solve_button = QPushButton("Solve")
        self.solve_button.clicked.connect(self.solve_problem)
        layout.addWidget(self.solve_button)
        
        layout.addStretch()
        group.setLayout(layout)
        return group
    
    def create_plot_panel(self):
        """Create the right plot panel with 4 subplots."""
        # Create figure with 4 subplots
        self.fig = Figure(figsize=(9, 8), dpi=100)
        self.ax1 = self.fig.add_subplot(2, 2, 1)  # State trajectory
        self.ax2 = self.fig.add_subplot(2, 2, 2)  # Control
        self.ax3 = self.fig.add_subplot(2, 2, 3)  # Costate
        self.ax4 = self.fig.add_subplot(2, 2, 4)  # Phase portrait
        
        self.canvas = FigureCanvas(self.fig)
        
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        widget.setLayout(layout)
        
        return widget
    
    def solve_problem(self):
        """Solve the optimal control problem and update plots."""
        try:
            # Get parameters
            x0 = [self.x0_spin.value(), self.v0_spin.value()]
            xf = [self.xf_spin.value(), self.vf_spin.value()]
            T = self.T_spin.value()
            umax = self.umax_spin.value()
            
            # Solve
            problem = DoubleIntegratorProblem(x0, xf, T, umax)
            self.solution = problem.solve()
            
            # Update plots
            self.update_plots()
        except Exception as e:
            print(f"Error solving problem: {e}")
            import traceback
            traceback.print_exc()
    
    def update_plots(self):
        """Update all plots with solution."""
        if self.solution is None:
            return
        
        t = self.solution['t']
        x = self.solution['x']
        u = self.solution['u']
        lam = self.solution['lambda']
        
        # Clear all axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Plot 1: State trajectory
        self.ax1.plot(t, x[:, 0], 'b-', label='Position', linewidth=2)
        self.ax1.plot(t, x[:, 1], 'r-', label='Velocity', linewidth=2)
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('State')
        self.ax1.set_title('State Trajectory')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Plot 2: Control
        self.ax2.plot(t, u, 'g-', linewidth=2)
        self.ax2.axhline(self.umax_spin.value(), color='k', linestyle='--', alpha=0.5)
        self.ax2.axhline(-self.umax_spin.value(), color='k', linestyle='--', alpha=0.5)
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Control u(t)')
        self.ax2.set_title('Optimal Control')
        self.ax2.grid(True, alpha=0.3)
        
        # Plot 3: Costate
        self.ax3.plot(t, lam[:, 0], 'b-', label='λ₁', linewidth=2)
        self.ax3.plot(t, lam[:, 1], 'r-', label='λ₂', linewidth=2)
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Costate')
        self.ax3.set_title('Costate Trajectory')
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
        
        # Plot 4: Phase portrait
        self.ax4.plot(x[:, 0], x[:, 1], 'b-', linewidth=2)
        self.ax4.plot(x[0, 0], x[0, 1], 'go', markersize=10, label='Start')
        self.ax4.plot(x[-1, 0], x[-1, 1], 'ro', markersize=10, label='End')
        self.ax4.set_xlabel('Position')
        self.ax4.set_ylabel('Velocity')
        self.ax4.set_title('Phase Portrait')
        self.ax4.legend()
        self.ax4.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()


def main():
    app = __import__('PyQt5.QtWidgets', fromlist=['QApplication']).QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
