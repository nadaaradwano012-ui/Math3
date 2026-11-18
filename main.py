import tkinter as tk
from tkinter import messagebox
import math

from methods import (
    bisection,
    modified_secant,
    gauss_seidel,
    linear_regression,
    lagrange_interpolation,
    true_error
)


# ==========================================
# 1. COLORS AND FONTS - REFERENCE DESIGN
# ==========================================

BG_MAIN = "#0f172a"
BG_SIDEBAR = "#1e293b"
CARD_COLOR = "#334155"
DISABLED_BG = "#1e293b"
TEXT_COLOR = "#f8fafc"
ACCENT_COLOR = "#a855f7"
ACTIVE_BTN = "#c084fc"

FONT_TITLE = ("Helvetica", 18, "bold")
FONT_H2 = ("Helvetica", 13, "bold")
FONT_LABEL = ("Helvetica", 11)
FONT_ENTRY = ("Helvetica", 11)


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def parse_function(expression):
    safe_env = {
        key: value for key, value in math.__dict__.items()
        if not key.startswith("__")
    }
    safe_env.update({"abs": abs, "pow": pow})
    expression = expression.replace("^", "**")

    return lambda x: eval(
        expression, {"__builtins__": {}}, {**safe_env, "x": x}
    )


def parse_numbers(value):
    return [float(item.strip()) for item in value.replace(",", " ").split()]


def get_stopping_params(stop_var, entries):
    if stop_var.get() == "tol":
        tolerance = float(entries["tol"].get())
        if tolerance <= 0:
            raise ValueError("Tolerance must be greater than zero.")
        return tolerance, 10000

    iterations = int(entries["max_iter"].get())
    if iterations <= 0:
        raise ValueError("Maximum iterations must be greater than zero.")
    return None, iterations


def format_error(approximate, exact):
    absolute, percentage = true_error(approximate, exact)
    if percentage is None:
        return f"{absolute:.6e} (absolute; exact value is zero)"
    return f"{absolute:.6e} absolute | {percentage:.6e}%"


def estimated_true_error(approximate, reference):
    """Estimated true error using a high-accuracy numerical result as reference."""
    absolute, percentage = true_error(approximate, reference)
    if percentage is None:
        return f"Estimated True Error: {absolute:.6e} absolute"
    return f"Estimated True Error: {absolute:.6e} absolute | {percentage:.6e}%"


def known_data_true_error(approximate, target_x, x_values, y_values):
    """Calculate true error automatically only when actual Y is in the supplied data."""
    for x_value, actual_y in zip(x_values, y_values):
        if math.isclose(target_x, x_value, rel_tol=1e-9, abs_tol=1e-12):
            return "True Error (known data point): " + format_error(approximate, actual_y)

    return (
        "True Error: Cannot be calculated automatically because actual Y "
        "at this X is unknown."
    )


def show_error(error):
    messagebox.showerror("Input Error", str(error))


# ==========================================
# 3. BISECTION METHOD
# ==========================================

def run_bisection(entries, stop_var, result_label):
    try:
        f = parse_function(entries["func"].get())
        a, b = float(entries["a"].get()), float(entries["b"].get())
        tolerance, iterations = get_stopping_params(stop_var, entries)

        (root, count, approx_error, residual), (
            best_root, best_count, best_approx, lowest_error
        ) = bisection(f, a, b, tolerance, iterations)

        approx_text = "N/A (first iteration)" if approx_error is None else f"{approx_error:.6e}%"
        best_approx_text = "N/A" if best_approx is None else f"{best_approx:.6e}%"

        output = (
            "[ Requested Result ]\n"
            f"Root: {root:.8f}   |   Iterations: {count}\n"
            f"Approx Error: {approx_text}   |   Error |f(root)|: {residual:.6e}\n"
            f"{estimated_true_error(root, best_root)}\n\n"
            "[ Best Root (Highest Accuracy) ]\n"
            f"Best Root: {best_root:.10f}   |   Iterations: {best_count}\n"
            f"Approx Error: {best_approx_text}   |   Lowest Residual: {lowest_error:.6e}"
        )
        result_label.config(text=output)

    except Exception as error:
        show_error(error)


# ==========================================
# 4. MODIFIED SECANT METHOD
# ==========================================

def run_modified_secant(entries, stop_var, result_label):
    try:
        f = parse_function(entries["func"].get())
        x0, delta = float(entries["x0"].get()), float(entries["delta"].get())
        tolerance, iterations = get_stopping_params(stop_var, entries)

        (root, count, approx_error, residual), (
            best_root, best_count, best_approx, lowest_error
        ) = modified_secant(f, x0, delta, tolerance, iterations)

        approx_text = "N/A" if approx_error is None else f"{approx_error:.6e}%"
        best_approx_text = "N/A" if best_approx is None else f"{best_approx:.6e}%"

        output = (
            "[ Requested Result ]\n"
            f"Root: {root:.8f}   |   Iterations: {count}\n"
            f"Approx Error: {approx_text}   |   Error |f(root)|: {residual:.6e}\n"
            f"{estimated_true_error(root, best_root)}\n\n"
            "[ Best Root (Highest Accuracy) ]\n"
            f"Best Root: {best_root:.10f}   |   Iterations: {best_count}\n"
            f"Approx Error: {best_approx_text}   |   Lowest Residual: {lowest_error:.6e}"
        )
        result_label.config(text=output)

    except Exception as error:
        show_error(error)


# ==========================================
# 5. GAUSS-SEIDEL METHOD
# ==========================================

def run_gauss_seidel(entries, stop_var, result_label):
    try:
        raw_rows = entries["matrix"].get("1.0", tk.END).strip().splitlines()
        matrix = [parse_numbers(row) for row in raw_rows if row.strip()]
        vector = parse_numbers(entries["vector"].get())
        tolerance, iterations = get_stopping_params(stop_var, entries)

        (solution, count, approx_error), (
            best_solution, best_count, lowest_error
        ) = gauss_seidel(matrix, vector, tolerance, iterations)

        solution_text = ", ".join(
            f"x{i + 1}={value:.6f}" for i, value in enumerate(solution)
        )
        best_text = ", ".join(
            f"x{i + 1}={value:.8f}" for i, value in enumerate(best_solution)
        )

        estimated_errors = [
            f"x{i + 1}: {estimated_true_error(solution[i], best_solution[i])}"
            for i in range(len(solution))
        ]

        output = (
            "[ Requested Result ]\n"
            f"Iterations: {count}   |   Approx Error: {approx_error:.6e}%\n"
            f"{solution_text}\n"
            "Estimated True Error (vs high-accuracy solution):\n"
            + "\n".join(estimated_errors)
            + "\n\n[ Best Solution (Highest Accuracy) ]\n"
            + f"Iterations: {best_count}   |   Lowest Approx Error: {lowest_error:.6e}%\n"
            + best_text
        )
        result_label.config(text=output)

    except Exception as error:
        show_error(error)


# ==========================================
# 6. LINEAR REGRESSION
# ==========================================

def run_linear_regression(entries, result_label):
    try:
        x = parse_numbers(entries["x_vals"].get())
        y = parse_numbers(entries["y_vals"].get())
        new_x = float(entries["new_x"].get())

        a0, a1, r2, predicted_y, rmse = linear_regression(x, y, new_x)
        sign = "+" if a1 >= 0 else "-"

        output = (
            f"Regression Line: y = {a0:.6f} {sign} {abs(a1):.6f}x\n"
            f"Intercept (a0): {a0:.6f}   |   Slope (a1): {a1:.6f}\n"
            f"R-squared (R²): {r2:.6f}\n"
            f"Predicted Y at X = {new_x}: {predicted_y:.8f}\n"
            f"RMSE: {rmse:.6e}\n"
            "Approx Error: N/A (direct, non-iterative method)\n"
            f"{known_data_true_error(predicted_y, new_x, x, y)}" 
        )
        result_label.config(text=output)

    except Exception as error:
        show_error(error)


# ==========================================
# 7. LAGRANGE INTERPOLATION
# ==========================================

def run_lagrange(entries, result_label):
    try:
        x = parse_numbers(entries["x_vals"].get())
        y = parse_numbers(entries["y_vals"].get())
        value = float(entries["value"].get())
        answer = lagrange_interpolation(x, y, value)

        output = (
            f"Interpolated Value at X = {value}:\n"
            f"f({value}) = {answer:.8f}\n"
            "Approx Error: N/A (direct, non-iterative method)\n"
            f"{known_data_true_error(answer, value, x, y)}" 
        )
        result_label.config(text=output)

    except Exception as error:
        show_error(error)


# ==========================================
# 8. NAVIGATION
# ==========================================

def show_frame(name):
    for frame in frames.values():
        frame.place_forget()

    frames[name].place(relwidth=1, relheight=1)

    for key, button in sidebar_buttons.items():
        if key == name:
            button.config(bg=ACTIVE_BTN, fg=BG_MAIN)
        else:
            button.config(bg=BG_SIDEBAR, fg=TEXT_COLOR)


# ==========================================
# 9. INPUT FIELD DESIGN
# ==========================================

def create_entry(parent, label_text, row, default_value=""):
    tk.Label(
        parent,
        text=label_text,
        bg=BG_MAIN,
        fg=TEXT_COLOR,
        font=FONT_LABEL
    ).grid(row=row, column=0, sticky="w", pady=8, padx=10)

    entry = tk.Entry(
        parent,
        bg=CARD_COLOR,
        fg=TEXT_COLOR,
        insertbackground=TEXT_COLOR,
        font=FONT_ENTRY,
        borderwidth=0,
        highlightthickness=1,
        highlightcolor=ACCENT_COLOR,
        highlightbackground="#94a3b8",
        disabledbackground=DISABLED_BG,
        disabledforeground="#64748b"
    )
    entry.grid(row=row, column=1, pady=8, padx=10, ipady=5, ipadx=5, sticky="ew")

    if default_value:
        entry.insert(0, default_value)

    return entry


# ==========================================
# 10. STOP CONDITION - ITERATIVE METHODS
# ==========================================

def add_stopping_controls(parent, start_row, entries):
    stop_var = tk.StringVar(value="iter")

    tk.Label(
        parent,
        text="Stop Condition:",
        bg=BG_MAIN,
        fg=ACCENT_COLOR,
        font=FONT_H2
    ).grid(row=start_row, column=0, sticky="w", pady=(12, 4), padx=10)

    radio_frame = tk.Frame(parent, bg=BG_MAIN)
    radio_frame.grid(row=start_row, column=1, sticky="w", pady=(12, 4), padx=10)

    entries["tol"] = create_entry(parent, "Tolerance Error (%):", start_row + 1, "0.0001")
    entries["max_iter"] = create_entry(parent, "Max Iterations:", start_row + 2, "5")

    def toggle_fields():
        if stop_var.get() == "tol":
            entries["tol"].config(state="normal")
            entries["max_iter"].config(state="disabled")
        else:
            entries["tol"].config(state="disabled")
            entries["max_iter"].config(state="normal")

    tk.Radiobutton(
        radio_frame, text="Tolerance Error",
        variable=stop_var, value="tol", command=toggle_fields,
        bg=BG_MAIN, fg=TEXT_COLOR, selectcolor=BG_SIDEBAR,
        activebackground=BG_MAIN, activeforeground=ACCENT_COLOR,
        font=FONT_LABEL
    ).pack(side="left", padx=(0, 15))

    tk.Radiobutton(
        radio_frame, text="Max Iterations",
        variable=stop_var, value="iter", command=toggle_fields,
        bg=BG_MAIN, fg=TEXT_COLOR, selectcolor=BG_SIDEBAR,
        activebackground=BG_MAIN, activeforeground=ACCENT_COLOR,
        font=FONT_LABEL
    ).pack(side="left")

    toggle_fields()
    return stop_var


# ==========================================
# 11. RESULTS BOX AND CALCULATE BUTTON
# ==========================================

def add_calculate_and_results(frame, command):
    result_label = tk.Label(
        frame,
        text="Results will appear here...",
        bg=CARD_COLOR,
        fg=TEXT_COLOR,
        font=("Helvetica", 10, "bold"),
        justify="left",
        anchor="w",
        padx=20,
        pady=16,
        wraplength=650
    )

    tk.Button(
        frame,
        text="Calculate",
        bg=ACCENT_COLOR,
        fg="white",
        activebackground=ACTIVE_BTN,
        activeforeground="white",
        font=FONT_H2,
        borderwidth=0,
        cursor="hand2",
        command=lambda: command(result_label)
    ).pack(pady=16, ipadx=40, ipady=8)

    result_label.pack(fill="x", padx=40)
    return result_label


# ==========================================
# 12. ITERATIVE METHOD PAGE
# ==========================================

def build_iterative_page(name, title, fields, callback):
    frame = tk.Frame(main_content, bg=BG_MAIN)
    frames[name] = frame

    tk.Label(
        frame, text=title, bg=BG_MAIN,
        fg=ACCENT_COLOR, font=FONT_TITLE
    ).pack(pady=20)

    form = tk.Frame(frame, bg=BG_MAIN)
    form.pack(pady=5, padx=40, fill="x")
    form.columnconfigure(1, weight=1)

    entries = {}
    for row, (key, (label, default)) in enumerate(fields.items()):
        entries[key] = create_entry(form, label, row, default)

    stop_var = add_stopping_controls(form, len(fields), entries)

    add_calculate_and_results(
        frame,
        lambda result: callback(entries, stop_var, result)
    )


# ==========================================
# 13. GAUSS-SEIDEL PAGE
# ==========================================

def build_gauss_page():
    frame = tk.Frame(main_content, bg=BG_MAIN)
    frames["gauss"] = frame

    tk.Label(
        frame, text="Gauss-Seidel Method",
        bg=BG_MAIN, fg=ACCENT_COLOR, font=FONT_TITLE
    ).pack(pady=20)

    form = tk.Frame(frame, bg=BG_MAIN)
    form.pack(pady=5, padx=40, fill="x")
    form.columnconfigure(1, weight=1)

    tk.Label(
        form, text="Matrix A (rows on new lines):",
        bg=BG_MAIN, fg=TEXT_COLOR, font=FONT_LABEL
    ).grid(row=0, column=0, sticky="nw", pady=8, padx=10)

    matrix = tk.Text(
        form, height=3, bg=CARD_COLOR, fg=TEXT_COLOR,
        insertbackground=TEXT_COLOR, font=FONT_ENTRY,
        borderwidth=0, highlightthickness=1,
        highlightcolor=ACCENT_COLOR, highlightbackground="#94a3b8"
    )
    matrix.grid(row=0, column=1, pady=8, padx=10, sticky="ew")
    matrix.insert("1.0", "4, -1, 0\n-1, 4, -1\n0, -1, 4")

    entries = {"matrix": matrix}
    entries["vector"] = create_entry(form, "Vector b (comma-separated):", 1, "15, 10, 10")

    stop_var = add_stopping_controls(form, 2, entries)

    add_calculate_and_results(
        frame,
        lambda result: run_gauss_seidel(entries, stop_var, result)
    )


# ==========================================
# 14. DIRECT METHOD PAGE
# ==========================================

def build_direct_page(name, title, fields, callback):
    frame = tk.Frame(main_content, bg=BG_MAIN)
    frames[name] = frame

    tk.Label(
        frame, text=title, bg=BG_MAIN,
        fg=ACCENT_COLOR, font=FONT_TITLE
    ).pack(pady=20)

    form = tk.Frame(frame, bg=BG_MAIN)
    form.pack(pady=10, padx=40, fill="x")
    form.columnconfigure(1, weight=1)

    entries = {}
    for row, (key, (label, default)) in enumerate(fields.items()):
        entries[key] = create_entry(form, label, row, default)

    add_calculate_and_results(
        frame,
        lambda result: callback(entries, result)
    )


# ==========================================
# 15. MAIN WINDOW
# ==========================================

def run():
    global root, sidebar, main_content, frames, sidebar_buttons

    root = tk.Tk()
    root.title("Numerical Analysis Project")
    root.geometry("1000x740")
    root.minsize(900, 680)
    root.configure(bg=BG_MAIN)

    sidebar = tk.Frame(root, bg=BG_SIDEBAR, width=230)
    sidebar.pack(side="left", fill="y")
    sidebar.pack_propagate(False)

    main_content = tk.Frame(root, bg=BG_MAIN)
    main_content.pack(side="right", fill="both", expand=True)

    frames = {}
    sidebar_buttons = {}

    # Bisection
    build_iterative_page(
        "bisection",
        "Bisection Method",
        {
            "func": ("Function f(x):", "x**3 - x - 2"),
            "a": ("Lower Bound (a):", "1"),
            "b": ("Upper Bound (b):", "2")
        },
        run_bisection
    )

    # Modified Secant
    build_iterative_page(
        "mod_secant",
        "Modified Secant Method",
        {
            "func": ("Function f(x):", "x**3 - x - 2"),
            "x0": ("Initial Guess (x0):", "1.5"),
            "delta": ("Perturbation (delta):", "0.01")
        },
        run_modified_secant
    )

    # Gauss-Seidel
    build_gauss_page()

    # Linear Regression
    build_direct_page(
        "regression",
        "Linear Regression",
        {
            "x_vals": ("X Values (comma-separated):", "1, 2, 3, 4"),
            "y_vals": ("Y Values (comma-separated):", "2, 4, 6, 8"),
            "new_x": ("Predict Y for X =", "5")
        },
        run_linear_regression
    )

    # Lagrange Interpolation
    build_direct_page(
        "lagrange",
        "Lagrange Interpolation",
        {
            "x_vals": ("X Values (comma-separated):", "1, 2, 4"),
            "y_vals": ("Y Values (comma-separated):", "1, 4, 16"),
            "value": ("Interpolate at X =", "3")
        },
        run_lagrange
    )

    # Sidebar title
    tk.Label(
        sidebar,
        text="Numerical\nMethods",
        bg=BG_SIDEBAR,
        fg=ACCENT_COLOR,
        font=("Helvetica", 21, "bold")
    ).pack(pady=30)

    # Sidebar method buttons
    menu_items = [
        ("Bisection", "bisection"),
        ("Modified Secant", "mod_secant"),
        ("Gauss-Seidel", "gauss"),
        ("Linear Regression", "regression"),
        ("Lagrange Interpolation", "lagrange")
    ]

    for label, key in menu_items:
        button = tk.Button(
            sidebar,
            text=label,
            bg=BG_SIDEBAR,
            fg=TEXT_COLOR,
            font=FONT_LABEL,
            borderwidth=0,
            activebackground=ACTIVE_BTN,
            activeforeground=BG_MAIN,
            cursor="hand2",
            command=lambda page=key: show_frame(page)
        )
        button.pack(fill="x", pady=5, ipady=10)
        sidebar_buttons[key] = button

    show_frame("bisection")
    root.mainloop()


if __name__ == "__main__":
    run()
