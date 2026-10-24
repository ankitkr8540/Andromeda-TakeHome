"""Compare one saved model's predictions with the benchmark reference function."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from function_approximation.data.dataset import rich_periodic
from function_approximation.inference import load_predictor

ROOT = Path(__file__).resolve().parent


def main():
    """Display a single-model comparison for user-supplied input values."""
    st.set_page_config(page_title="Compare predictions", layout="centered")
    st.title("Compare predictions")
    st.write(
        "Choose a saved model and enter x values or generate a range to compare its predictions "
        "with the reference function."
    )
    # Only offer completed runs with both artifacts required by inference.
    runs = sorted(
        checkpoint.parent.parent
        for checkpoint in (ROOT / "outputs/best").glob("*/*/*/model/best_model.pt")
        if (checkpoint.parent / "preprocessor.joblib").is_file()
    )
    if not runs:
        st.info(
            "No saved models found. Train a model using a config in configs/best/ "
            "first, then refresh this page."
        )
        return
    run = st.selectbox(
        "Saved model",
        runs,
        format_func=lambda path: str(path.relative_to(ROOT / "outputs/best")),
    )
    mode = st.radio(
        "Input method", ["Enter numbers", "Generate a range"], horizontal=True
    )
    with st.form("inputs"):
        if mode == "Enter numbers":
            raw = st.text_area(
                "Input x values",
                "-3.14, 0, 100",
                help="Separate numbers with commas, spaces, or newlines.",
            )
        else:
            start = st.number_input("Start x", value=-31.42)
            end = st.number_input("End x", value=31.42)
            count = st.number_input(
                "Number of points", min_value=2, max_value=10000, value=500, step=1
            )
            st.caption("Generates evenly spaced inputs, including both endpoints.")
        view = st.radio("Show comparison as", ["Table", "Plot"], horizontal=True)
        submitted = st.form_submit_button("Compare")
    st.caption(
        "Training interval: −10π to 10π (about −31.42 to 31.42). "
        "Inputs outside this interval test extrapolation."
    )
    if not submitted:
        return
    try:
        if mode == "Generate a range":
            if not np.isfinite([start, end]).all() or start >= end:
                raise ValueError(
                    "Start and end must be finite, with start less than end."
                )
            x = np.linspace(start, end, int(count))
        else:
            tokens = re.split(r"[\s,]+", raw.strip())
            if not raw.strip() or len(tokens) > 10000:
                raise ValueError("Enter between 1 and 10,000 numbers.")
            x = np.array([float(token) for token in tokens])
        if not np.isfinite(x).all():
            raise ValueError("Use finite numbers only.")
    except ValueError as error:
        st.error(f"Invalid inputs: {error}")
        return
    try:
        predicted = load_predictor(run).predict(x)
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        st.error(f"Could not load or evaluate this saved model: {error}")
        return
    # Reference values are for comparison only; the predictor never uses them.
    reference = rich_periodic(x)
    table = pd.DataFrame(
        {
            "x": x,
            "Reference output": reference,
            "Predicted output": predicted,
            "Absolute error": np.abs(reference - predicted),
        }
    )
    if view == "Table":
        st.dataframe(table, hide_index=True, width="stretch")
    else:
        fig, ax = plt.subplots()
        if mode == "Generate a range":
            ax.plot(x, reference, label="Reference", color="gray")
            ax.plot(x, predicted, label="Prediction", color="#cf238a", linestyle="--")
        else:
            ax.scatter(x, reference, label="Reference", color="gray", marker="o")
            ax.scatter(x, predicted, label="Prediction", color="#cf238a", marker="x")
        ax.set(xlabel="Input x", ylabel="Output", title="Reference vs prediction")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Curves connect predictions at the generated inputs. Increase the point count for finer detail."
            if mode == "Generate a range"
            else "Each point represents an entered input; no curve is inferred between inputs."
        )
    st.download_button(
        "Download comparison CSV",
        table.to_csv(index=False),
        "predictions.csv",
        "text/csv",
    )


if __name__ == "__main__":
    main()
