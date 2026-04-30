import sys

import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.solver import DoubleIntegratorProblem


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PMP Visualizer - Minimum Time Double Integrator")
        self.setGeometry(100, 100, 1600, 900)

        self.optimal_solution = None
        self.perturbed_solution = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.create_control_panel(), 1)
        main_layout.addWidget(self.create_plot_panel(), 3)
        central_widget.setLayout(main_layout)

        self.on_perturbation_changed()
        self.solve_optimal()

    def create_control_panel(self):
        """Create the left control panel."""
        main_group = QGroupBox("Minimum-Time Problem")
        layout = QVBoxLayout()

        problem_text = """
        <b>Double Integrator PMP</b><br><br>
        <b>Dynamics:</b><br>
        x_dot = v<br>
        v_dot = u<br><br>
        <b>Constraint:</b><br>
        |u| <= u_max<br><br>
        <b>Cost:</b><br>
        J = integral 1 dt = T<br><br>
        <b>PMP Structure:</b><br>
        H = 1 + p1 v + p2 u = 0<br>
        u*(t) is bang-bang and switches when p2(t) = 0.
        """

        problem_label = QLabel(problem_text)
        problem_label.setWordWrap(True)
        problem_label.setStyleSheet(
            "border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9;"
        )
        layout.addWidget(problem_label)

        layout.addWidget(QLabel("Problem Parameters"))

        layout.addWidget(QLabel("Initial Position (x0):"))
        self.x0_spin = self.create_spin_box(-10.0, 10.0, 0.0, 0.1)
        layout.addWidget(self.x0_spin)

        layout.addWidget(QLabel("Initial Velocity (v0):"))
        self.v0_spin = self.create_spin_box(-5.0, 5.0, 0.0, 0.1)
        layout.addWidget(self.v0_spin)

        layout.addWidget(QLabel("Target Position (xf):"))
        self.xf_spin = self.create_spin_box(-10.0, 10.0, 5.0, 0.1)
        layout.addWidget(self.xf_spin)

        layout.addWidget(QLabel("Target Velocity (vf):"))
        self.vf_spin = self.create_spin_box(-5.0, 5.0, 0.0, 0.1)
        layout.addWidget(self.vf_spin)

        layout.addWidget(QLabel("Initial Time Guess (s):"))
        self.T_spin = self.create_spin_box(0.0, 20.0, 5.0, 0.1)
        layout.addWidget(self.T_spin)

        layout.addWidget(QLabel("Control Bound (u_max):"))
        self.umax_spin = self.create_spin_box(0.1, 10.0, 1.0, 0.1)
        layout.addWidget(self.umax_spin)

        self.solve_button = QPushButton("Solve Minimum Time Problem")
        self.solve_button.clicked.connect(self.solve_optimal)
        layout.addWidget(self.solve_button)

        layout.addWidget(QLabel("PMP Perturbation"))

        self.perturbation_enabled = QCheckBox("Compare perturbed control at T*")
        self.perturbation_enabled.setChecked(False)
        self.perturbation_enabled.stateChanged.connect(self.on_perturbation_changed)
        layout.addWidget(self.perturbation_enabled)

        layout.addWidget(QLabel("Perturbation Type:"))
        self.perturbation_type = QComboBox()
        self.perturbation_type.addItems(["Control Noise", "Control Bias"])
        layout.addWidget(self.perturbation_type)

        layout.addWidget(QLabel("Perturbation Strength:"))
        self.perturbation_strength = self.create_spin_box(0.0, 1.0, 0.1, 0.01, decimals=3)
        layout.addWidget(self.perturbation_strength)

        self.update_perturbation_button = QPushButton("Apply Perturbation")
        self.update_perturbation_button.clicked.connect(self.apply_perturbation)
        layout.addWidget(self.update_perturbation_button)

        layout.addWidget(QLabel("Results"))

        layout.addWidget(QLabel("Minimum Time T*:"))
        self.optimal_time_label = QLabel("-")
        layout.addWidget(self.optimal_time_label)

        layout.addWidget(QLabel("Switch Time:"))
        self.switch_time_label = QLabel("-")
        layout.addWidget(self.switch_time_label)

        layout.addWidget(QLabel("Optimal Arc Sequence:"))
        self.arc_sequence_label = QLabel("-")
        self.arc_sequence_label.setWordWrap(True)
        layout.addWidget(self.arc_sequence_label)

        layout.addWidget(QLabel("Optimal Terminal Error:"))
        self.optimal_error_label = QLabel("-")
        layout.addWidget(self.optimal_error_label)

        layout.addWidget(QLabel("Perturbed Terminal Error at T*:"))
        self.perturbed_error_label = QLabel("-")
        layout.addWidget(self.perturbed_error_label)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()
        main_group.setLayout(layout)
        return main_group

    def create_plot_panel(self):
        """Create the right plot panel with six subplots."""
        self.fig = Figure(figsize=(11, 8), dpi=100)
        self.ax1 = self.fig.add_subplot(2, 3, 1)
        self.ax2 = self.fig.add_subplot(2, 3, 2)
        self.ax3 = self.fig.add_subplot(2, 3, 3)
        self.ax4 = self.fig.add_subplot(2, 3, 4)
        self.ax5 = self.fig.add_subplot(2, 3, 5)
        self.ax6 = self.fig.add_subplot(2, 3, 6)

        self.canvas = FigureCanvas(self.fig)

        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        widget.setLayout(layout)
        return widget

    def create_spin_box(self, minimum, maximum, value, step, decimals=3):
        spin_box = QDoubleSpinBox()
        spin_box.setRange(minimum, maximum)
        spin_box.setValue(value)
        spin_box.setSingleStep(step)
        spin_box.setDecimals(decimals)
        return spin_box

    def on_perturbation_changed(self):
        """Enable or disable perturbation inputs."""
        enabled = self.perturbation_enabled.isChecked()
        self.perturbation_type.setEnabled(enabled)
        self.perturbation_strength.setEnabled(enabled)
        self.update_perturbation_button.setEnabled(enabled and self.optimal_solution is not None)

        if not enabled:
            self.perturbed_solution = None
            if hasattr(self, "perturbed_error_label"):
                self.perturbed_error_label.setText("-")
            if self.optimal_solution is not None:
                self.update_plots()

    def solve_optimal(self):
        """Solve the minimum-time double-integrator problem."""
        try:
            problem = self.create_problem()
            self.optimal_solution = problem.solve()
            self.perturbed_solution = None

            optimal_time = self.optimal_solution["T"]
            self.optimal_time_label.setText(f"{optimal_time:.6g} s")
            self.switch_time_label.setText(self.format_switch_time(self.optimal_solution["switch_time"]))
            self.arc_sequence_label.setText(self.optimal_solution["arc_sequence"])
            self.optimal_error_label.setText(
                f"{self.optimal_solution['terminal_error']:.3e}"
            )
            self.perturbed_error_label.setText("-")
            self.status_label.setText("Solved with free final time.")
            self.T_spin.setValue(optimal_time)
            self.update_perturbation_button.setEnabled(self.perturbation_enabled.isChecked())

            self.update_plots()
        except Exception as exc:
            self.status_label.setText(f"Solve failed: {exc}")
            QMessageBox.warning(self, "Solve failed", str(exc))

    def apply_perturbation(self):
        """Simulate a perturbed version of the optimal control over the optimal time."""
        if self.optimal_solution is None:
            self.status_label.setText("Solve the minimum-time problem first.")
            return

        try:
            optimal_time = self.optimal_solution["T"]
            if np.isclose(optimal_time, 0.0):
                self.perturbed_solution = None
                self.perturbed_error_label.setText("0.000e+00")
                self.status_label.setText("The boundary conditions are already satisfied at T* = 0.")
                self.update_plots()
                return

            perturbation_type = self.perturbation_type.currentText()
            strength = self.perturbation_strength.value()
            u_opt = self.optimal_solution["u"]

            if perturbation_type == "Control Noise":
                perturbation = np.random.normal(0.0, strength, len(u_opt))
            else:
                phase = np.linspace(0.0, 2.0 * np.pi, len(u_opt))
                perturbation = strength * np.sin(phase)

            umax = self.umax_spin.value()
            u_perturbed = np.clip(u_opt + perturbation, -umax, umax)

            problem = self.create_problem(time_override=optimal_time)
            final_index = len(u_perturbed) - 1

            def perturbed_control(t):
                index = min(int(t / optimal_time * len(u_perturbed)), final_index)
                return u_perturbed[index]

            self.perturbed_solution = problem.simulate_with_control(perturbed_control)
            terminal_error = self.perturbed_solution["terminal_error"]
            self.perturbed_error_label.setText(f"{terminal_error:.3e}")
            self.status_label.setText("Perturbed control simulated over the optimal horizon T*.")
            self.update_plots()
        except Exception as exc:
            self.status_label.setText(f"Perturbation failed: {exc}")
            QMessageBox.warning(self, "Perturbation failed", str(exc))

    def create_problem(self, time_override=None):
        x0 = [self.x0_spin.value(), self.v0_spin.value()]
        xf = [self.xf_spin.value(), self.vf_spin.value()]
        T = self.T_spin.value() if time_override is None else time_override
        umax = self.umax_spin.value()
        return DoubleIntegratorProblem(x0, xf, T, umax)

    def format_switch_time(self, switch_time):
        if switch_time is None:
            return "No switch"
        return f"{switch_time:.6g} s"

    def update_plots(self):
        """Update all plots with the current optimal and perturbed solutions."""
        if self.optimal_solution is None:
            return

        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
            ax.clear()

        t_opt = self.optimal_solution["t"]
        x_opt = self.optimal_solution["x"]
        u_opt = self.optimal_solution["u"]
        p_opt = self.optimal_solution["p"]
        H_opt = self.optimal_solution["H"]

        self.plot_state_trajectory(t_opt, x_opt)
        self.plot_control(t_opt, u_opt)
        self.plot_phase_portrait(x_opt)
        self.plot_costate(t_opt, p_opt)
        self.plot_hamiltonian(t_opt, H_opt)
        self.plot_terminal_error()

        self.fig.tight_layout()
        self.canvas.draw()

    def plot_state_trajectory(self, t_opt, x_opt):
        self.ax1.plot(t_opt, x_opt[:, 0], "b-", label="Position", linewidth=2)
        self.ax1.plot(t_opt, x_opt[:, 1], "r-", label="Velocity", linewidth=2)

        if self.perturbed_solution is not None:
            t_pert = self.perturbed_solution["t"]
            x_pert = self.perturbed_solution["x"]
            self.ax1.plot(t_pert, x_pert[:, 0], "b--", label="Position perturbed", linewidth=1.5)
            self.ax1.plot(t_pert, x_pert[:, 1], "r--", label="Velocity perturbed", linewidth=1.5)

        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("State")
        self.ax1.set_title("State Trajectory")
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)

    def plot_control(self, t_opt, u_opt):
        self.ax2.step(t_opt, u_opt, "g-", linewidth=2.5, where="post", label="Optimal control")

        if self.perturbed_solution is not None:
            self.ax2.plot(
                self.perturbed_solution["t"],
                self.perturbed_solution["u"],
                color="orange",
                linewidth=1.5,
                linestyle="--",
                label="Perturbed control",
            )

        umax = self.umax_spin.value()
        self.ax2.axhline(umax, color="k", linestyle="--", alpha=0.3)
        self.ax2.axhline(-umax, color="k", linestyle="--", alpha=0.3)
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("u(t)")
        self.ax2.set_title("Bang-Bang Control")
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)

    def plot_phase_portrait(self, x_opt):
        self.ax3.plot(x_opt[:, 0], x_opt[:, 1], "b-", linewidth=2, label="Optimal")
        self.ax3.plot(x_opt[0, 0], x_opt[0, 1], "go", markersize=9, label="Start")
        self.ax3.plot(x_opt[-1, 0], x_opt[-1, 1], "ro", markersize=9, label="Target")

        if self.perturbed_solution is not None:
            x_pert = self.perturbed_solution["x"]
            self.ax3.plot(
                x_pert[:, 0],
                x_pert[:, 1],
                color="orange",
                linewidth=1.5,
                linestyle="--",
                label="Perturbed",
            )
            self.ax3.plot(x_pert[-1, 0], x_pert[-1, 1], "x", color="orange", markersize=9)

        self.ax3.set_xlabel("Position")
        self.ax3.set_ylabel("Velocity")
        self.ax3.set_title("Phase Portrait")
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)

    def plot_costate(self, t_opt, p_opt):
        self.ax4.plot(t_opt, p_opt[:, 0], "b-", label="p1", linewidth=2)
        self.ax4.plot(t_opt, p_opt[:, 1], "r-", label="p2", linewidth=2)
        switch_time = self.optimal_solution.get("switch_time")
        if switch_time is not None:
            self.ax4.axvline(switch_time, color="k", linestyle="--", alpha=0.25)
        self.ax4.axhline(0.0, color="k", linestyle=":", alpha=0.4)
        self.ax4.set_xlabel("Time (s)")
        self.ax4.set_ylabel("Costate")
        self.ax4.set_title("Costate")
        self.ax4.legend()
        self.ax4.grid(True, alpha=0.3)

    def plot_hamiltonian(self, t_opt, H_opt):
        self.ax5.plot(t_opt, H_opt, color="purple", linewidth=2)
        self.ax5.axhline(0.0, color="k", linestyle="--", alpha=0.35)
        self.ax5.set_xlabel("Time (s)")
        self.ax5.set_ylabel("H(t)")
        self.ax5.set_title("Hamiltonian")
        self.ax5.grid(True, alpha=0.3)

    def plot_terminal_error(self):
        optimal_error = self.optimal_solution["terminal_error"]

        if self.perturbed_solution is None:
            self.ax6.bar(["Optimal"], [optimal_error], color=["green"], alpha=0.75)
            self.ax6.set_title("Terminal Error at T*")
            self.ax6.set_ylabel("||x(T*) - xf||")
            self.ax6.grid(True, alpha=0.3, axis="y")
            return

        perturbed_error = self.perturbed_solution["terminal_error"]
        labels = ["Optimal", "Perturbed"]
        errors = [optimal_error, perturbed_error]
        bars = self.ax6.bar(labels, errors, color=["green", "orange"], alpha=0.75, edgecolor="black")

        for bar, error in zip(bars, errors):
            self.ax6.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{error:.2e}",
                ha="center",
                va="bottom",
            )

        self.ax6.set_title("Terminal Error at T*")
        self.ax6.set_ylabel("||x(T*) - xf||")
        self.ax6.grid(True, alpha=0.3, axis="y")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
