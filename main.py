import os
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np

import KS

st.set_page_config(
    page_title="Density and Speed Data", page_icon=":bar_chart:", layout="wide"
)
# Base directory where all directories are located
BASE_DIR = "2ColData"


def load_directories(base_dir: str) -> List[str]:
    """Function to load all directories within the base directory"""
    return [
        dir
        for dir in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, dir))
    ]


def load_data_from_dir(directory: str) -> pd.DataFrame:
    """Function to load all files data from a directory"""
    dfs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # assuming files are txt
            df = pd.read_csv(
                os.path.join(directory, filename),
                sep="\t",
                comment="#",
                names=["rho", "velocity"],
            )
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def plot_data(
    data: Dict[str, pd.DataFrame],
    data2: pd.DataFrame,
    freq: int,
    dx: float,
    N: int,
) -> go.Figure:
    """Function to plot data using Plotly"""
    fig = go.Figure()
    for dir, df in data.items():
        x_values = np.arange(0, df["rho"].max() + dx, dx)
        fig.add_trace(
            go.Scatter(
                x=df["rho"][::freq],
                y=df["velocity"][::freq],
                mode="markers",
                name=dir,
                marker=dict(opacity=0.1),
            )
        )
        v10, v50, v90 = KS.percentiles(df, dx=dx, N=N)
        fig.add_trace(
            # ax.plot(x_values[:len(v10_values)], v10_values, label='V10[x]')
            go.Scatter(x=x_values[: len(v10)], y=v10, mode="lines", name="V10[x]")
        )
        fig.add_trace(
            go.Scatter(x=x_values[: len(v50)], y=v50, mode="lines", name="V50[x]")
        )
        fig.add_trace(
            go.Scatter(x=x_values[: len(v90)], y=v90, mode="lines", name="V90[x]")
        )

    if not data2.empty:
        fig.add_trace(
            go.Scatter(
                x=data2["rho"][::freq],
                y=data2["velocity"][::freq],
                mode="markers",
                name="Reference data",
                marker=dict(symbol="cross", opacity=0.5, size=5, color="red"),
            )
        )
    fig.update_layout(xaxis_title="Density / 1/m", yaxis_title="Speed / m/s")
    return fig


def compare_data(data: Dict[str, pd.DataFrame], data2: pd.DataFrame):
    """
    compare two data clouds
    data and data2 are dataframes with two columns rho and velocity
    """

    rho_list = []
    velocity_list = []
    for _, df in data.items():
        rho_list.append(df["rho"])
        velocity_list.append(df["velocity"])

    rho_list = pd.concat(rho_list)
    velocity_list = pd.concat(velocity_list)
    return KS.CDFDistance(
        rho_list, velocity_list, list(data2["rho"]), list(data2["velocity"])
    )


if __name__ == "__main__":
    st.title("Directory Data Visualization")
    # ================================== Interface
    c1, c2, c3 = st.columns(3)
    frequency = c1.number_input(
        "Enter the frequency of the points to be plotted",
        min_value=1,
        value=10,
        help="The lower the slower",
    )
    N = c2.number_input(
        "N",
        min_value=5,
        value=50,
        help="The minimal data points to consider in calculation of confidence interval",
    )
    N = int(N)
    dx = c3.number_input(
        "dx",
        min_value=0.1,
        value=0.5,
        help="Density discritisation of density for calculation of confidence interval",
    )
    dx = float(dx)
    do_KS_test = c1.checkbox(
        "Make KS-test?",
        help="Kolmogorov-Smirnov test may be slow, depending on the amount of data",
    )
    st.write("-----------")
    # ==================================
    c1, c2 = st.columns((0.5, 0.5))
    directories: List[str] = load_directories(BASE_DIR)
    directories.sort()
    c1.header("Experiments")
    frequency = int(frequency)
    selected_directories = [dir for dir in directories if c1.checkbox(f"{dir}")]
    data = {}
    for directory in selected_directories:
        data[directory] = load_data_from_dir(os.path.join(BASE_DIR, directory))

    data_to_compare = pd.DataFrame()
    if do_KS_test:
        compare_directory = c2.selectbox(
            "Kolmogorov-Smirnov Test  (0 is perfect match!)",
            directories,
            help="Choose data to compare to the selected data from the left column",
        )
        data_to_compare = load_data_from_dir(os.path.join(BASE_DIR, compare_directory))
        if data:
            result = compare_data(data, data_to_compare)
            c2.info(f"Distance: {result:.2f}")

    fig = plot_data(data, data_to_compare, frequency, dx, N)
    c2.plotly_chart(fig)
