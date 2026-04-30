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

from core.solver import (
    ClassicalLQDoubleIntegratorProblem,
    DoubleIntegratorProblem,
    MinimumEnergyDoubleIntegratorProblem,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PMP Visualizer - Minimum Time Double Integrator")
        self.setGeometry(100, 100, 1600, 900)

        self.optimal_solution = None
        self.perturbed_solution = None
        self.current_problem = None
        self.last_problem_mode = "minimum_time"

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
        self.problem_group = QGroupBox("PMP Problem")
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Problem Type:"))
        self.problem_type_combo = QComboBox()
        self.problem_type_combo.addItem("Minimum Time (Bang-Bang)", "minimum_time")
        self.problem_type_combo.addItem("Minimum Energy (Smooth)", "minimum_energy")
        self.problem_type_combo.addItem("Classical LQ PMP (Unconstrained)", "classical_lq")
        self.problem_type_combo.currentIndexChanged.connect(self.on_problem_type_changed)
        layout.addWidget(self.problem_type_combo)

        self.problem_label = QLabel(self.problem_description_text())
        self.problem_label.setWordWrap(True)
        self.problem_label.setStyleSheet(
            "border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9;"
        )
        layout.addWidget(self.problem_label)

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

        self.time_label = QLabel("Initial Time Guess (s):")
        layout.addWidget(self.time_label)
        self.T_spin = self.create_spin_box(0.0, 20.0, 5.0, 0.1, decimals=6)
        layout.addWidget(self.T_spin)

        self.input_constraint_checkbox = QCheckBox("Constrain input |u| <= u_max")
        self.input_constraint_checkbox.setChecked(False)
        self.input_constraint_checkbox.stateChanged.connect(self.on_input_constraint_changed)
        layout.addWidget(self.input_constraint_checkbox)

        self.umax_label = QLabel("Control Bound (u_max):")
        layout.addWidget(self.umax_label)
        self.umax_spin = self.create_spin_box(0.1, 10.0, 1.0, 0.1)
        layout.addWidget(self.umax_spin)

        self.solve_button = QPushButton("Solve Minimum Time Problem")
        self.solve_button.clicked.connect(self.solve_optimal)
        layout.addWidget(self.solve_button)

        self.variation_title_label = QLabel("Switch-Time Variation")
        layout.addWidget(self.variation_title_label)

        self.perturbation_enabled = QCheckBox("Show modified switch-time trajectory")
        self.perturbation_enabled.setChecked(True)
        self.perturbation_enabled.stateChanged.connect(self.on_perturbation_changed)
        layout.addWidget(self.perturbation_enabled)

        self.switch_variation_label = QLabel("Modified Switch Time (s):")
        layout.addWidget(self.switch_variation_label)
        self.perturbed_switch_time_spin = self.create_spin_box(0.0, 20.0, 0.0, 0.01, decimals=8)
        self.perturbed_switch_time_spin.valueChanged.connect(self.update_switch_time_perturbation)
        layout.addWidget(self.perturbed_switch_time_spin)

        self.multiplier_variation_label = QLabel("Perturb Control Multiplier:")
        layout.addWidget(self.multiplier_variation_label)
        self.control_multiplier_spin = self.create_spin_box(-2.0, 3.0, 1.0, 0.01, decimals=4)
        self.control_multiplier_spin.valueChanged.connect(self.update_control_multiplier_variation)
        layout.addWidget(self.control_multiplier_spin)

        layout.addWidget(QLabel("Results"))

        self.primary_time_caption = QLabel("Minimum Time T*:")
        layout.addWidget(self.primary_time_caption)
        self.optimal_time_label = QLabel("-")
        layout.addWidget(self.optimal_time_label)

        self.secondary_result_caption = QLabel("Switch Time:")
        layout.addWidget(self.secondary_result_caption)
        self.switch_time_label = QLabel("-")
        layout.addWidget(self.switch_time_label)

        self.arc_sequence_caption = QLabel("Optimal Arc Sequence:")
        layout.addWidget(self.arc_sequence_caption)
        self.arc_sequence_label = QLabel("-")
        self.arc_sequence_label.setWordWrap(True)
        layout.addWidget(self.arc_sequence_label)

        self.optimal_error_caption = QLabel("Optimal Terminal Error:")
        layout.addWidget(self.optimal_error_caption)
        self.optimal_error_label = QLabel("-")
        layout.addWidget(self.optimal_error_label)

        self.perturbed_error_caption = QLabel("Perturbed Terminal Error at T*:")
        layout.addWidget(self.perturbed_error_caption)
        self.perturbed_error_label = QLabel("-")
        layout.addWidget(self.perturbed_error_label)

        self.variation_energy_caption = QLabel("Scaled Energy Cost J:")
        layout.addWidget(self.variation_energy_caption)
        self.variation_energy_label = QLabel("-")
        layout.addWidget(self.variation_energy_label)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.problem_group.setLayout(layout)
        self.configure_problem_mode()
        return self.problem_group

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

    def current_problem_mode(self):
        if not hasattr(self, "problem_type_combo"):
            return "minimum_time"
        return self.problem_type_combo.currentData()

    def is_minimum_time_mode(self):
        return self.current_problem_mode() == "minimum_time"

    def is_minimum_energy_mode(self):
        return self.current_problem_mode() == "minimum_energy"

    def is_classical_lq_mode(self):
        return self.current_problem_mode() == "classical_lq"

    def problem_description_text(self):
        if self.is_classical_lq_mode():
            return """
            <b>Classical Unconstrained LQ PMP</b><br><br>
            <b>Dynamics:</b><br>
            x_dot = v<br>
            v_dot = u<br><br>
            <b>Fixed Horizon and Endpoints:</b><br>
            x(0), v(0), x(T), v(T) are prescribed<br><br>
            <b>Cost:</b><br>
            J = integral 0.5 (x^2 + u^2) dt<br><br>
            <b>PMP Structure:</b><br>
            H = 0.5 (x^2 + u^2) + p1 v + p2 u<br>
            u*(t) = -p2(t), with p_dot = -H_x.<br><br>
            <b>Variation:</b><br>
            Scale p(0) and re-integrate the Hamiltonian system.
            """

        if self.is_minimum_energy_mode():
            return """
            <b>Minimum-Energy Double Integrator PMP</b><br><br>
            <b>Dynamics:</b><br>
            x_dot = v<br>
            v_dot = u<br><br>
            <b>Fixed Horizon:</b><br>
            T is prescribed<br><br>
            <b>Optional Constraint:</b><br>
            |u| <= u_max<br><br>
            <b>Cost:</b><br>
            J = integral 0.5 u^2 dt<br><br>
            <b>PMP Structure:</b><br>
            H = 0.5 u^2 + p1 v + p2 u<br>
            u*(t) = clip(-p2(t), -u_max, u_max) when bounded.
            """

        return """
        <b>Minimum-Time Double Integrator PMP</b><br><br>
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

    def on_problem_type_changed(self):
        new_mode = self.current_problem_mode()
        if new_mode in {"minimum_energy", "classical_lq"} and self.last_problem_mode == "minimum_time":
            if self.T_spin.value() < 5.0:
                self.T_spin.setValue(5.0)

        self.optimal_solution = None
        self.perturbed_solution = None
        self.configure_problem_mode()
        self.solve_optimal()
        self.last_problem_mode = new_mode

    def input_constraint_enabled(self):
        return (
            hasattr(self, "input_constraint_checkbox")
            and self.is_minimum_energy_mode()
            and self.input_constraint_checkbox.isChecked()
        )

    def on_input_constraint_changed(self):
        self.configure_problem_mode()
        if self.optimal_solution is not None:
            self.solve_optimal()

    def configure_problem_mode(self):
        minimum_time = self.is_minimum_time_mode()
        minimum_energy = self.is_minimum_energy_mode()
        classical_lq = self.is_classical_lq_mode()
        self.setWindowTitle(
            "PMP Visualizer - Minimum Time Double Integrator"
            if minimum_time
            else (
                "PMP Visualizer - Minimum Energy Double Integrator"
                if minimum_energy
                else "PMP Visualizer - Classical LQ Double Integrator"
            )
        )
        self.problem_group.setTitle(
            "Minimum-Time Problem"
            if minimum_time
            else ("Minimum-Energy Problem" if minimum_energy else "Classical LQ PMP Problem")
        )
        self.problem_label.setText(self.problem_description_text())
        self.time_label.setText("Initial Time Guess (s):" if minimum_time else "Fixed Time Horizon T (s):")
        self.solve_button.setText(
            "Solve Minimum Time Problem"
            if minimum_time
            else ("Solve Minimum Energy Problem" if minimum_energy else "Solve Classical LQ PMP")
        )

        self.T_spin.blockSignals(True)
        if minimum_time:
            self.T_spin.setRange(0.0, 20.0)
        else:
            self.T_spin.setRange(0.1, 20.0)
            if self.T_spin.value() <= 0.0:
                self.T_spin.setValue(5.0)
        self.T_spin.blockSignals(False)

        show_input_bound = minimum_time or self.input_constraint_enabled()
        self.input_constraint_checkbox.setVisible(minimum_energy)
        self.umax_label.setVisible(show_input_bound)
        self.umax_spin.setVisible(show_input_bound)

        self.variation_title_label.setText(
            "Switch-Time Variation"
            if minimum_time
            else ("Control-Multiplier Variation" if minimum_energy else "Initial-Costate Variation")
        )
        self.perturbation_enabled.setText(
            "Show modified switch-time trajectory"
            if minimum_time
            else (
                "Show scaled-control trajectory"
                if minimum_energy
                else "Show scaled-costate trajectory"
            )
        )

        self.variation_title_label.setVisible(True)
        self.perturbation_enabled.setVisible(True)
        self.switch_variation_label.setVisible(minimum_time)
        self.perturbed_switch_time_spin.setVisible(minimum_time)
        self.multiplier_variation_label.setText(
            "Perturb Control Multiplier:" if minimum_energy else "Initial Costate Multiplier:"
        )
        self.multiplier_variation_label.setVisible(minimum_energy or classical_lq)
        self.control_multiplier_spin.setVisible(minimum_energy or classical_lq)

        self.primary_time_caption.setText("Minimum Time T*:" if minimum_time else "Fixed Time T:")
        self.secondary_result_caption.setText(
            "Switch Time:"
            if minimum_time
            else ("Energy Cost J:" if minimum_energy else "Cost Functional J:")
        )
        self.arc_sequence_caption.setText(
            "Optimal Arc Sequence:" if minimum_time else "Optimal Control:"
        )
        self.perturbed_error_caption.setText(
            "Perturbed Terminal Error at T*:"
            if minimum_time
            else "Scaled Terminal Error at T:"
        )
        self.perturbed_error_caption.setVisible(True)
        self.perturbed_error_label.setVisible(True)
        self.variation_energy_caption.setText(
            "Scaled Energy Cost J:" if minimum_energy else "Scaled Cost Functional J:"
        )
        self.variation_energy_caption.setVisible(minimum_energy or classical_lq)
        self.variation_energy_label.setVisible(minimum_energy or classical_lq)
        self.configure_variation_controls()

    def configure_variation_controls(self):
        if self.is_minimum_time_mode():
            self.configure_switch_time_control()
        elif self.is_minimum_energy_mode() or self.is_classical_lq_mode():
            self.configure_control_multiplier_control()
        else:
            if hasattr(self, "perturbed_switch_time_spin"):
                self.perturbed_switch_time_spin.setEnabled(False)
            if hasattr(self, "control_multiplier_spin"):
                self.control_multiplier_spin.setEnabled(False)

    def update_active_variation(self):
        if self.is_minimum_time_mode():
            self.update_switch_time_perturbation()
        elif self.is_minimum_energy_mode() or self.is_classical_lq_mode():
            self.update_control_multiplier_variation()

    def active_control_bound(self):
        if self.is_minimum_time_mode():
            return self.umax_spin.value()
        if self.optimal_solution is None:
            return None
        return self.optimal_solution.get("control_bound")

    def on_perturbation_changed(self):
        """Enable or disable perturbation inputs."""
        enabled = self.perturbation_enabled.isChecked()
        self.configure_variation_controls()

        if not enabled:
            self.perturbed_solution = None
            if hasattr(self, "perturbed_error_label"):
                self.perturbed_error_label.setText("-")
            if hasattr(self, "variation_energy_label"):
                self.variation_energy_label.setText("-")
            if self.optimal_solution is not None:
                self.update_plots()
        elif self.optimal_solution is not None:
            self.update_active_variation()

    def solve_optimal(self):
        """Solve the selected double-integrator PMP problem."""
        try:
            self.current_problem = self.create_problem()
            self.optimal_solution = self.current_problem.solve()
            self.perturbed_solution = None

            horizon = self.optimal_solution["T"]
            self.optimal_time_label.setText(f"{horizon:.6g} s")
            if self.is_minimum_time_mode():
                self.switch_time_label.setText(self.format_switch_time(self.optimal_solution["switch_time"]))
            else:
                self.switch_time_label.setText(f"{self.optimal_solution['cost']:.6g}")

            self.arc_sequence_label.setText(self.optimal_solution["arc_sequence"])
            self.optimal_error_label.setText(
                f"{self.optimal_solution['terminal_error']:.3e}"
            )
            self.perturbed_error_label.setText("-")
            self.variation_energy_label.setText("-")
            self.status_label.setText(
                "Solved with free final time."
                if self.is_minimum_time_mode()
                else (
                    "Solved with fixed final time."
                    if self.is_minimum_energy_mode()
                    else "Solved classical Hamiltonian boundary-value problem."
                )
            )

            if self.is_minimum_time_mode():
                self.T_spin.setValue(horizon)
            self.configure_variation_controls()

            if self.perturbation_enabled.isChecked():
                self.update_active_variation()
            else:
                self.update_plots()
        except Exception as exc:
            self.status_label.setText(f"Solve failed: {exc}")
            QMessageBox.warning(self, "Solve failed", str(exc))

    def update_switch_time_perturbation(self):
        """Simulate the bang-bang control with the selected switch time."""
        if self.optimal_solution is None:
            self.status_label.setText("Solve the minimum-time problem first.")
            return
        if not self.perturbation_enabled.isChecked() or not self.is_minimum_time_mode():
            return

        try:
            optimal_time = self.optimal_solution["T"]
            if np.isclose(optimal_time, 0.0):
                self.perturbed_solution = None
                self.perturbed_error_label.setText("0.000e+00")
                self.status_label.setText("The boundary conditions are already satisfied at T* = 0.")
                self.update_plots()
                return

            u_opt = self.optimal_solution["u"]
            first_control = float(u_opt[0])
            second_control = -first_control
            switch_time = np.clip(self.perturbed_switch_time_spin.value(), 0.0, optimal_time)

            self.perturbed_solution = self.simulate_switch_time_control(
                switch_time,
                first_control,
                second_control,
                optimal_time,
            )
            self.add_perturbed_pmp_quantities(switch_time, first_control)
            self.perturbed_solution["switch_time"] = switch_time
            terminal_error = self.perturbed_solution["terminal_error"]
            self.perturbed_error_label.setText(f"{terminal_error:.3e}")
            self.status_label.setText("Modified switch-time trajectory simulated at T*.")
            self.update_plots()
        except Exception as exc:
            self.status_label.setText(f"Perturbation failed: {exc}")
            QMessageBox.warning(self, "Perturbation failed", str(exc))

    def update_control_multiplier_variation(self):
        """Simulate the selected smooth-mode multiplier variation."""
        if self.optimal_solution is None:
            self.status_label.setText("Solve the selected fixed-time problem first.")
            return
        if not self.perturbation_enabled.isChecked() or self.is_minimum_time_mode():
            return

        try:
            if self.current_problem is None:
                self.current_problem = self.create_problem()

            multiplier = self.control_multiplier_spin.value()
            self.perturbed_solution = self.current_problem.build_multiplier_solution(
                self.optimal_solution,
                multiplier,
            )
            terminal_error = self.perturbed_solution["terminal_error"]
            self.perturbed_error_label.setText(f"{terminal_error:.3e}")
            self.variation_energy_label.setText(f"{self.perturbed_solution['cost']:.6g}")
            self.status_label.setText(
                "Scaled-control trajectory simulated at fixed T."
                if self.is_minimum_energy_mode()
                else "Scaled initial-costate trajectory simulated at fixed T."
            )
            self.update_plots()
        except Exception as exc:
            self.status_label.setText(f"Control multiplier failed: {exc}")
            QMessageBox.warning(self, "Control multiplier failed", str(exc))

    def create_problem(self, time_override=None):
        x0 = [self.x0_spin.value(), self.v0_spin.value()]
        xf = [self.xf_spin.value(), self.vf_spin.value()]
        T = self.T_spin.value() if time_override is None else time_override
        umax = self.umax_spin.value()
        if self.is_minimum_time_mode():
            return DoubleIntegratorProblem(x0, xf, T, umax)
        if self.is_minimum_energy_mode():
            control_bound = umax if self.input_constraint_enabled() else None
            return MinimumEnergyDoubleIntegratorProblem(x0, xf, T, control_bound)
        return ClassicalLQDoubleIntegratorProblem(x0, xf, T)

    def simulate_switch_time_control(self, switch_time, first_control, second_control, horizon):
        """Analytically simulate a one-switch bang-bang control."""
        t = np.linspace(0.0, horizon, 500)
        x0 = self.x0_spin.value()
        v0 = self.v0_spin.value()

        u = np.where(t <= switch_time, first_control, second_control)
        x = np.zeros_like(t)
        v = np.zeros_like(t)

        first_arc = t <= switch_time
        x[first_arc] = x0 + v0 * t[first_arc] + 0.5 * first_control * t[first_arc]**2
        v[first_arc] = v0 + first_control * t[first_arc]

        if not np.all(first_arc):
            tau = t[~first_arc] - switch_time
            x_switch = x0 + v0 * switch_time + 0.5 * first_control * switch_time**2
            v_switch = v0 + first_control * switch_time
            x[~first_arc] = x_switch + v_switch * tau + 0.5 * second_control * tau**2
            v[~first_arc] = v_switch + second_control * tau

        target = np.array([self.xf_spin.value(), self.vf_spin.value()])
        terminal_state = np.array([x[-1], v[-1]])

        return {
            "t": t,
            "x": np.column_stack([x, v]),
            "u": u,
            "cost": horizon,
            "T": horizon,
            "terminal_error": float(np.linalg.norm(terminal_state - target)),
        }

    def configure_switch_time_control(self):
        if not hasattr(self, "perturbed_switch_time_spin"):
            return

        enabled = (
            self.is_minimum_time_mode()
            and self.perturbation_enabled.isChecked()
            and self.optimal_solution is not None
        )
        if not enabled:
            self.perturbed_switch_time_spin.setEnabled(False)
            return

        optimal_time = self.optimal_solution["T"]
        has_time_to_switch = not np.isclose(optimal_time, 0.0)
        nominal_switch = self.optimal_solution["switch_time"]
        if nominal_switch is None:
            nominal_switch = optimal_time

        self.perturbed_switch_time_spin.blockSignals(True)
        self.perturbed_switch_time_spin.setRange(0.0, max(optimal_time, 0.0))
        self.perturbed_switch_time_spin.setSingleStep(max(optimal_time / 200.0, 0.001))
        self.perturbed_switch_time_spin.setValue(nominal_switch)
        self.perturbed_switch_time_spin.blockSignals(False)
        self.perturbed_switch_time_spin.setEnabled(has_time_to_switch)

    def configure_control_multiplier_control(self):
        if not hasattr(self, "control_multiplier_spin"):
            return

        enabled = (
            (self.is_minimum_energy_mode() or self.is_classical_lq_mode())
            and self.perturbation_enabled.isChecked()
            and self.optimal_solution is not None
        )
        self.control_multiplier_spin.setEnabled(enabled)
        if not enabled:
            return

        self.control_multiplier_spin.blockSignals(True)
        self.control_multiplier_spin.setValue(1.0)
        self.control_multiplier_spin.blockSignals(False)

    def add_perturbed_pmp_quantities(self, switch_time, first_control):
        """Build the costate and Hamiltonian for the selected switch time."""
        if self.optimal_solution is None or self.perturbed_solution is None:
            return

        t_pert = self.perturbed_solution["t"]
        switch_velocity = self.v0_spin.value() + first_control * switch_time

        if np.isclose(switch_velocity, 0.0):
            p1_pert = np.zeros_like(t_pert)
            p2_pert = np.zeros_like(t_pert)
        else:
            p1_value = -1.0 / switch_velocity
            p1_pert = np.full_like(t_pert, p1_value)
            p2_pert = p1_value * (switch_time - t_pert)

        p_pert = np.column_stack([p1_pert, p2_pert])

        velocity = self.perturbed_solution["x"][:, 1]
        control = self.perturbed_solution["u"]
        H_pert = 1.0 + p1_pert * velocity + p2_pert * control

        self.perturbed_solution["p"] = p_pert
        self.perturbed_solution["H"] = H_pert

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
        if self.is_minimum_time_mode():
            self.ax2.step(t_opt, u_opt, "g-", linewidth=2.5, where="post", label="Optimal control")
        else:
            self.ax2.plot(t_opt, u_opt, "g-", linewidth=2.5, label="Optimal control")

        if self.perturbed_solution is not None:
            self.ax2.plot(
                self.perturbed_solution["t"],
                self.perturbed_solution["u"],
                color="orange",
                linewidth=1.5,
                linestyle="--",
                label="Perturbed control",
            )
            switch_time = self.perturbed_solution.get("switch_time")
            if switch_time is not None:
                self.ax2.axvline(switch_time, color="orange", linestyle=":", alpha=0.65)

        control_bound = self.active_control_bound()
        if control_bound is not None:
            umax = control_bound
            self.ax2.axhline(umax, color="k", linestyle="--", alpha=0.3)
            self.ax2.axhline(-umax, color="k", linestyle="--", alpha=0.3)
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("u(t)")
        self.ax2.set_title(
            "Bang-Bang Control"
            if self.is_minimum_time_mode()
            else ("Smooth Control" if self.is_minimum_energy_mode() else "Unconstrained Control")
        )
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

        if self.perturbed_solution is not None and "p" in self.perturbed_solution:
            t_pert = self.perturbed_solution["t"]
            p_pert = self.perturbed_solution["p"]
            self.ax4.plot(
                t_pert,
                p_pert[:, 0],
                color="navy",
                linestyle="--",
                label="p1 perturbed",
                linewidth=1.5,
            )
            self.ax4.plot(
                t_pert,
                p_pert[:, 1],
                color="darkred",
                linestyle="--",
                label="p2 perturbed",
                linewidth=1.5,
            )

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
        self.ax5.plot(t_opt, H_opt, color="purple", linewidth=2, label="Optimal")

        if self.perturbed_solution is not None and "H" in self.perturbed_solution:
            self.ax5.plot(
                self.perturbed_solution["t"],
                self.perturbed_solution["H"],
                color="orange",
                linestyle="--",
                linewidth=1.7,
                label="Perturbed",
            )

        if self.is_minimum_time_mode():
            self.ax5.axhline(0.0, color="k", linestyle="--", alpha=0.35)
        else:
            self.ax5.axhline(np.mean(H_opt), color="k", linestyle="--", alpha=0.35)
        self.ax5.set_xlabel("Time (s)")
        self.ax5.set_ylabel("H(t)")
        self.ax5.set_title("Hamiltonian")
        self.ax5.legend()
        self.ax5.grid(True, alpha=0.3)

    def plot_terminal_error(self):
        if not self.is_minimum_time_mode():
            self.plot_energy_comparison()
            return

        optimal_error = self.optimal_solution["terminal_error"]
        title = "Terminal Error at T*" if self.is_minimum_time_mode() else "Terminal Error at T"
        ylabel = "||x(T*) - xf||" if self.is_minimum_time_mode() else "||x(T) - xf||"

        if self.perturbed_solution is None:
            self.ax6.bar(["Optimal"], [optimal_error], color=["green"], alpha=0.75)
            self.ax6.set_title(title)
            self.ax6.set_ylabel(ylabel)
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

        self.ax6.set_title(title)
        self.ax6.set_ylabel(ylabel)
        self.ax6.grid(True, alpha=0.3, axis="y")

    def plot_energy_comparison(self):
        optimal_energy = self.optimal_solution["cost"]

        if self.perturbed_solution is None:
            labels = ["Optimal"]
            energies = [optimal_energy]
            colors = ["green"]
        else:
            labels = ["Optimal", "Scaled"]
            energies = [optimal_energy, self.perturbed_solution["cost"]]
            colors = ["green", "orange"]

        bars = self.ax6.bar(labels, energies, color=colors, alpha=0.75, edgecolor="black")

        for bar, energy in zip(bars, energies):
            self.ax6.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{energy:.3g}",
                ha="center",
                va="bottom",
            )

        if self.is_minimum_energy_mode():
            title = "Energy Comparison"
            ylabel = "J = integral 0.5 u^2 dt"
        else:
            title = "Cost Functional"
            ylabel = "J = integral 0.5 (x^2 + u^2) dt"

        self.ax6.set_title(title)
        self.ax6.set_ylabel(ylabel)
        self.ax6.grid(True, alpha=0.3, axis="y")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
