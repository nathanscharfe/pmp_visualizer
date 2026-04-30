import sys
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, 
    QGroupBox, QSlider, QSpinBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from core.solver import DoubleIntegratorProblem


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PMP Visualizer - Double Integrator")
        self.setGeometry(100, 100, 1400, 800)
        
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
        
        # Initialize solution
        self.solution = None
    
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
