import tkinter as tk
from tkinter import messagebox
import math
import sympy as sp
from methods import (
    bisection,
    modified_secant,
    seidel,
    linear_regression,
    lagrange_interpolation,
)

BG_COLOR = "#0f172a"
CARD_COLOR = "#1e293b"
TEXT_COLOR = "#f8fafc"
ACCENT_COLOR = "#a855f7"
GLOW_ACTIVE = "#c084fc"
BORDER_COLOR = "#334155"

FONT_TITLE = ("Helvetica", 18, "bold")
FONT_LABEL = ("Helvetica", 11)
FONT_ENTRY = ("Helvetica", 12)


def get_function(func_str):
    x = sp.symbols("x")
    expr = sp.sympify(func_str)
    return sp.lambdify(x, expr, "math"), expr, x


def solve_bisection(entries, lbl_result):
    try:
        equation = entries["func"].get()
        a = float(entries["a"].get())
        b = float(entries["b"].get())
        max_iter = int(entries["iter"].get())

        best_root, exact_root, min_error, true_error, actual_iter = bisection(equation, a, b, max_iter)

        if best_root is None:
            messagebox.showerror("Error", "Invalid interval or iterations (f(a) and f(b) must have opposite signs).")
            return

        res = f"Best Root: {best_root:.6f}\n\n"
        res += f"Iterations: {actual_iter}\n"

        if min_error is not None:
            res += f"Relative Approximate Error: {min_error:.4f}%\n"
        else:
            res += "Relative Approximate Error: N/A\n"

        if exact_root is not None:
            res += f"Exact Solution: {exact_root} ({float(exact_root):.6f})\n"
            if true_error is not None:
                res += f"Relative True Error: {true_error:.4f}%\n"
        else:
            res += "Exact Solution: Not available symbolically\n"

        lbl_result.config(text=res)
    except Exception:
        messagebox.showerror("Error", "Please check your inputs (e.g., Function: x**3 - x - 2).")


def solve_modified_secant(entries, lbl_result):
    try:
        func_str = entries["func"].get()
        f, expr, x_sym = get_function(func_str)
        x0 = float(entries["x0"].get())
        delta = float(entries["delta"].get())
        max_iter = int(entries["iter"].get())
        true_val_str = entries["true"].get().strip()

        if max_iter <= 0:
            raise ValueError

        result = modified_secant(f, x0, delta, max_iter)
        if result is None:
            messagebox.showerror("Error", "Division by zero encountered. Try a different initial guess.")
            return

        final_x, actual_iter, final_err, best_root, best_error = result

        exact_val = None
        exact_label = None
        if true_val_str:
            exact_val = float(true_val_str)
            exact_label = f"{exact_val:.6f}"
        else:
            try:
                real_roots = [float(r) for r in sp.solve(expr, x_sym) if r.is_real]
                if real_roots:
                    closest = min(real_roots, key=lambda r: abs(r - best_root))
                    exact_val = closest
                    exact_label = f"{closest:.6f}"
            except Exception:
                exact_val = None

        is_exact = (f(best_root) == 0) or (exact_val is not None and abs(best_root - exact_val) < 1e-12)

        res = f"Best Root: {best_root:.6f}\n\n"
        res += "Result: True Value (Exact)\n" if is_exact else "Result: Approximate Value\n"
        res += f"Iterations: {actual_iter}\n"
        res += f"Relative Approximate Error: {best_error:.4f}%\n"

        if exact_val is not None:
            res += f"Exact Solution: {exact_label}\n"
            rel_true = abs((exact_val - best_root) / exact_val) * 100 if exact_val != 0 else abs(exact_val - best_root) * 100
            res += f"Relative True Error: {rel_true:.4f}%\n"
        else:
            res += "Exact Solution: Not available\n"

        lbl_result.config(text=res)
    except Exception:
        messagebox.showerror("Error", "Please check your inputs (e.g., Function: x**3 - x - 2).")


def solve_gauss_seidel(entries, lbl_result):
    try:
        raw_rows = entries["matrix_a"].get("1.0", tk.END).strip().splitlines()
        A = [[float(num.strip()) for num in row.replace(",", " ").split()] for row in raw_rows if row.strip()]
        b = [float(num.strip()) for num in entries["vector_b"].get().replace(",", " ").split()]
        max_iter = int(entries["iter"].get())
        true_vec_str = entries["true"].get().strip()

        if len(A) == 0 or len(A) != len(b) or any(len(row) != len(b) for row in A) or max_iter <= 0:
            raise ValueError

        result = seidel(A, b, max_iter)
        if result is None:
            messagebox.showerror("Error", "Zero on the main diagonal (Division by zero).")
            return

        best_sol, actual_iter, approx_err, is_exact = result
        sol_text = ", ".join([f"x{i+1} = {val:.6f}" for i, val in enumerate(best_sol)])

        res = f"Best Solution: {sol_text}\n\n"
        res += "Result: True Value (Exact)\n" if is_exact else "Result: Approximate Value\n"
        res += f"Iterations: {actual_iter}\n"
        res += f"Relative Approximate Error: {approx_err:.4f}%\n"

        if true_vec_str:
            true_vec = [float(num.strip()) for num in true_vec_str.replace(",", " ").split()]
            if len(true_vec) != len(best_sol):
                messagebox.showerror("Error", "True Values count must match number of variables.")
                return
            rel_true = max(
                abs((t - c) / t) * 100 if t != 0 else abs(t - c) * 100
                for t, c in zip(true_vec, best_sol)
            )
            res += f"Relative True Error: {rel_true:.4f}%\n"

        lbl_result.config(text=res)
    except Exception:
        messagebox.showerror("Error", "Please check matrix, vector, and iterations inputs.")


def solve_linear_regression(entries, lbl_result):
    try:
        x = [float(i.strip()) for i in entries["x_vals"].get().split(",")]
        y = [float(i.strip()) for i in entries["y_vals"].get().split(",")]
        new_x = float(entries["new_x"].get())

        if len(x) != len(y) or len(x) == 0:
            messagebox.showerror("Error", "X and Y must have the same number of values.")
            return

        result = linear_regression(x, y, new_x)
        if result is None:
            messagebox.showerror("Error", "Denominator is zero (all X values are identical).")
            return

        a0, a1, r2, expected_y = result
        sign = "+" if a1 >= 0 else "-"

        res = f"Equation: y = {a0:.4f} {sign} {abs(a1):.4f}x\n\n"
        res += f"Intercept (a0): {a0:.4f}\n"
        res += f"Slope (a1): {a1:.4f}\n"
        res += f"R-squared (R²): {r2:.4f}\n"
        res += f"Expected Y (at X = {new_x}): {expected_y:.6f}"

        lbl_result.config(text=res)
    except Exception:
        messagebox.showerror("Error", "Please enter valid comma-separated numbers.")


def solve_lagrange(entries, lbl_result):
    try:
        x = [float(i.strip()) for i in entries["x_vals"].get().split(",")]
        y = [float(i.strip()) for i in entries["y_vals"].get().split(",")]
        target_x = float(entries["target_x"].get())

        if len(x) != len(y) or len(x) == 0:
            messagebox.showerror("Error", "X and Y must have the same number of values.")
            return

        result = lagrange_interpolation(x, y, target_x)
        if result is None:
            messagebox.showerror("Error", "Duplicate X values are not allowed.")
            return

        res = "Interpolated Value:\n\n"
        res += f"At X = {target_x}\n"
        res += f"Expected Y = {result:.6f}"

        lbl_result.config(text=res)
    except Exception:
        messagebox.showerror("Error", "Please enter valid comma-separated numbers.")


root = tk.Tk()
root.title("Numerical Methods Calculator")
root.geometry("760x660")
root.configure(bg=BG_COLOR)

sidebar = tk.Frame(root, bg=CARD_COLOR, width=210)
sidebar.pack(side="left", fill="y")
sidebar.pack_propagate(False)

separator = tk.Frame(root, bg=BORDER_COLOR, width=1)
separator.pack(side="left", fill="y")

main_area = tk.Frame(root, bg=BG_COLOR)
main_area.pack(side="right", fill="both", expand=True)

frames = {}
nav_buttons = {}


def switch_page(page_key):
    for frame in frames.values():
        frame.place_forget()
    frames[page_key].place(relwidth=1, relheight=1)

    for key, btn in nav_buttons.items():
        if key == page_key:
            btn.config(bg=ACCENT_COLOR, fg="white", font=("Helvetica", 11, "bold"))
        else:
            btn.config(bg=CARD_COLOR, fg=TEXT_COLOR, font=FONT_LABEL)


def create_input(parent, label_text, row):
    tk.Label(parent, text=label_text, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT_LABEL).grid(
        row=row, column=0, sticky="w", pady=10, padx=10
    )
    entry = tk.Entry(
        parent,
        bg=CARD_COLOR,
        fg=TEXT_COLOR,
        insertbackground=TEXT_COLOR,
        font=FONT_ENTRY,
        borderwidth=0,
        highlightthickness=1,
        highlightcolor=ACCENT_COLOR,
        highlightbackground=BORDER_COLOR,
    )
    entry.grid(row=row, column=1, pady=10, padx=10, ipady=5, ipadx=5, sticky="ew")
    return entry


def create_method_page(key, title, fields, btn_text, command_fn):
    page = tk.Frame(main_area, bg=BG_COLOR)
    frames[key] = page

    tk.Label(page, text=title, bg=BG_COLOR, fg=ACCENT_COLOR, font=FONT_TITLE).pack(pady=25)

    frame_inputs = tk.Frame(page, bg=BG_COLOR)
    frame_inputs.pack(pady=10, padx=35, fill="x")
    frame_inputs.columnconfigure(1, weight=1)

    entries = {}
    for row_idx, (field_key, label_text) in enumerate(fields):
        entries[field_key] = create_input(frame_inputs, label_text, row_idx)

    lbl_result = tk.Label(
        page,
        text="Results will appear here...",
        bg=CARD_COLOR,
        fg=ACCENT_COLOR,
        font=("Helvetica", 11, "bold"),
        justify="left",
        padx=20,
        pady=20,
        borderwidth=1,
        relief="ridge",
    )

    btn_calc = tk.Button(
        page,
        text=btn_text,
        bg=ACCENT_COLOR,
        fg="white",
        font=("Helvetica", 12, "bold"),
        borderwidth=0,
        activebackground=GLOW_ACTIVE,
        activeforeground="white",
        cursor="hand2",
        command=lambda: command_fn(entries, lbl_result),
    )
    btn_calc.pack(pady=25, ipadx=30, ipady=10)
    lbl_result.pack(fill="x", padx=40)


def create_gauss_page():
    page = tk.Frame(main_area, bg=BG_COLOR)
    frames["gauss"] = page

    tk.Label(page, text="Gauss-Seidel Calculator", bg=BG_COLOR, fg=ACCENT_COLOR, font=FONT_TITLE).pack(pady=25)

    frame_inputs = tk.Frame(page, bg=BG_COLOR)
    frame_inputs.pack(pady=10, padx=35, fill="x")
    frame_inputs.columnconfigure(1, weight=1)

    entries = {}

    tk.Label(frame_inputs, text="Matrix A (rows on new lines):", bg=BG_COLOR, fg=TEXT_COLOR, font=FONT_LABEL).grid(
        row=0, column=0, sticky="nw", pady=10, padx=10
    )
    txt_a = tk.Text(
        frame_inputs,
        height=3,
        bg=CARD_COLOR,
        fg=TEXT_COLOR,
        insertbackground=TEXT_COLOR,
        font=FONT_ENTRY,
        borderwidth=0,
        highlightthickness=1,
        highlightcolor=ACCENT_COLOR,
        highlightbackground=BORDER_COLOR,
    )
    txt_a.grid(row=0, column=1, pady=10, padx=10, sticky="ew")
    entries["matrix_a"] = txt_a

    entries["vector_b"] = create_input(frame_inputs, "Vector b (e.g., 15, 10, 10):", 1)
    entries["iter"] = create_input(frame_inputs, "Iterations (e.g., 10):", 2)
    entries["true"] = create_input(frame_inputs, "True Values (Optional):", 3)

    lbl_result = tk.Label(
        page,
        text="Results will appear here...",
        bg=CARD_COLOR,
        fg=ACCENT_COLOR,
        font=("Helvetica", 11, "bold"),
        justify="left",
        padx=20,
        pady=20,
        borderwidth=1,
        relief="ridge",
    )

    btn_calc = tk.Button(
        page,
        text="Calculate Solution",
        bg=ACCENT_COLOR,
        fg="white",
        font=("Helvetica", 12, "bold"),
        borderwidth=0,
        activebackground=GLOW_ACTIVE,
        activeforeground="white",
        cursor="hand2",
        command=lambda: solve_gauss_seidel(entries, lbl_result),
    )
    btn_calc.pack(pady=25, ipadx=30, ipady=10)
    lbl_result.pack(fill="x", padx=40)


create_method_page(
    "bisection",
    "Bisection Calculator",
    [
        ("func", "Function (e.g., x**3 - x - 2):"),
        ("a", "Lower Bound (a):"),
        ("b", "Upper Bound (b):"),
        ("iter", "Iterations (e.g., 10):"),
    ],
    "Calculate Root",
    solve_bisection,
)

create_method_page(
    "mod_secant",
    "Modified Secant Calculator",
    [
        ("func", "Function (e.g., x**3 - x - 2):"),
        ("x0", "Initial Guess (x0):"),
        ("delta", "Delta (e.g., 0.01):"),
        ("iter", "Iterations (e.g., 10):"),
        ("true", "True Value (Optional):"),
    ],
    "Calculate Root",
    solve_modified_secant,
)

create_gauss_page()

create_method_page(
    "regression",
    "Linear Regression Calculator",
    [
        ("x_vals", "X Values (e.g., 1, 2, 3):"),
        ("y_vals", "Y Values (e.g., 2, 4, 6):"),
        ("new_x", "Expected X (Predict Y):"),
    ],
    "Calculate Line & Predict",
    solve_linear_regression,
)

create_method_page(
    "lagrange",
    "Lagrange Interpolation",
    [
        ("x_vals", "X Values (e.g., 1, 2, 4):"),
        ("y_vals", "Y Values (e.g., 1, 4, 16):"),
        ("target_x", "Value of X to Interpolate:"),
    ],
    "Calculate Value",
    solve_lagrange,
)

tk.Label(sidebar, text="Math 3\nMethods", bg=CARD_COLOR, fg=ACCENT_COLOR, font=FONT_TITLE).pack(pady=30)

menu_items = [
    ("Bisection", "bisection"),
    ("Modified Secant", "mod_secant"),
    ("Gauss-Seidel", "gauss"),
    ("Linear Regression", "regression"),
    ("Lagrange Interpolation", "lagrange"),
]

for text, key in menu_items:
    btn = tk.Button(
        sidebar,
        text=text,
        bg=CARD_COLOR,
        fg=TEXT_COLOR,
        font=FONT_LABEL,
        borderwidth=0,
        activebackground=GLOW_ACTIVE,
        activeforeground="white",
        cursor="hand2",
        command=lambda k=key: switch_page(k),
    )
    btn.pack(fill="x", pady=4, ipady=10)
    nav_buttons[key] = btn

switch_page("bisection")
root.mainloop()