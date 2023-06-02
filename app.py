import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import KS
import docs

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
                sep="\s+",
                comment="#",
                names=["rho", "velocity"],
            )
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_data(filename) -> pd.DataFrame:
    """Function to load uploaded file data"""
    if filename:
        if filename.name.endswith(".txt"):  # assuming files are txt
            df = pd.read_csv(
                filename,
                sep="\t",
                comment="#",
                names=["rho", "velocity"],
            )
            return df
        else:
            st.error("Invalid file type. Please upload a '.txt' file.")
            return pd.DataFrame()
    else:
        st.info("Please upload a file.")
        return pd.DataFrame()


def plot_data(
    data: Dict[str, pd.DataFrame],
    reference_data: pd.DataFrame,
    v10: List[float],
    v50: List[float],
    v90: List[float],
    freq: int,
    dx: float,
) -> go.Figure:
    """Function to plot data using Plotly"""
    fig = go.Figure()

    for dir, df in data.items():
        print(dir)
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

        fig.add_trace(
            go.Scatter(x=x_values[: len(v10)], y=v10, mode="lines", name="V10[x]")
        )
        fig.add_trace(
            go.Scatter(x=x_values[: len(v50)], y=v50, mode="lines", name="V50[x]")
        )
        fig.add_trace(
            go.Scatter(x=x_values[: len(v90)], y=v90, mode="lines", name="V90[x]")
        )

    if not reference_data.empty:
        fig.add_trace(
            go.Scatter(
                x=reference_data["rho"][::freq],
                y=reference_data["velocity"][::freq],
                mode="markers",
                name="Reference data",
                marker=dict(symbol="cross", opacity=0.5, size=5, color="red"),
            )
        )

    fig.update_layout(xaxis_title="Density / 1/m", yaxis_title="Speed / m/s")
    return fig


def plot_data2(
    data: Dict[str, pd.DataFrame],
    reference_data: pd.DataFrame,
    v10,
    v50,
    v90,
    x_values,
    freq: int,
    dx: float,
):
    """Function to plot data using matplotlib"""
    fig = plt.figure()

    for dir, df in data.items():
        print(dir)
        plt.scatter(
            df["rho"][::freq],
            df["velocity"][::freq],
            label=dir,
            alpha=0.1,
        )
    if v10:
        plt.plot(
            x_values[: len(v10)], v10, "--", linewidth=2, color="k", label="V10[x]"
        )
    if v50:
        plt.plot(
            x_values[: len(v50)], v50, "-.", linewidth=2, color="k", label="V50[x]"
        )
    if v90:
        plt.plot(
            x_values[: len(v90)], v90, "-x", linewidth=2, color="k", label="V90[x]"
        )

    if not reference_data.empty:
        plt.plot(
            reference_data["rho"][::freq],
            reference_data["velocity"][::freq],
            "x",
            color="red",
            label="Reference data",
            alpha=0.7,
        )
    plt.xlabel("Density / 1/m")
    plt.ylabel("Velocity / m/s")
    plt.legend()
    return fig


def compare_data(data: Dict[str, pd.DataFrame], data2: pd.DataFrame) -> float:
    """
    compare two data clouds using KS-test
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
    tab1, tab2 = st.tabs(["Analysis", "References"])
    # ================================== Interface
    with tab2:
        docs.methods()
        st.divider()
        docs.references()
    with tab1:
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
            "KS-test",
            help="Kolmogorov-Smirnov test may be slow, depending on the amount of data",
        )
        do_percentiles = c2.checkbox(
            "Calculate percentiles",
        )
        upload_file = c3.checkbox(
            "Upload data",
            help="Data format: two columns. First column for density. Second column for speed",
        )
        st.divider()
        # ==================================

        # ==================================
        c11, c12 = st.columns((0.5, 0.5))
        directories: List[str] = load_directories(BASE_DIR)
        directories.sort()
        c11.header("Experiments")
        frequency = int(frequency)
        selected_directories = [dir for dir in directories if c11.checkbox(f"{dir}")]
        data = {}
        start_time = time.perf_counter()
        for directory in selected_directories:
            data[directory] = load_data_from_dir(os.path.join(BASE_DIR, directory))

        end_time = time.perf_counter()
        runtime = end_time - start_time
        print(f"Load_data: {runtime:.2f} seconds")

        data_to_compare = pd.DataFrame()
        uploaded_file = pd.DataFrame()
        if upload_file:
            uploaded_file = c3.file_uploader(
                "Upload your data file in txt format", type=["txt"]
            )
            if uploaded_file is not None:
                directories.insert(0, "Uploaded data")
                data_to_compare = load_data(uploaded_file)

        if do_KS_test:
            print("start KS")
            start_time = time.perf_counter()
            compare_directory = c12.selectbox(
                "Kolmogorov-Smirnov Test  (1 is perfect match!)",
                directories,
                help="Choose data to compare to the selected data from the left column",
            )
            if compare_directory != "Uploaded data":
                data_to_compare = load_data_from_dir(
                    os.path.join(BASE_DIR, compare_directory)
                )
            if data:
                result = 1 - compare_data(data, data_to_compare)
                c12.metric("Similarity", f"{result*100:.0f}%", f"{result:.2f}")
                end_time = time.perf_counter()
                runtime = end_time - start_time
                print(f"KS_test: {runtime:.2f} seconds")

        start_time = time.perf_counter()
        dfs = []
        for _, df in data.items():
            dfs.append(df)

        v10 = []
        v50 = []
        v90 = []
        x_values = []
        if dfs and do_percentiles:
            dfs = pd.concat(dfs, ignore_index=True)
            v10, v50, v90, x_values = KS.percentiles(dfs, dx=dx, N=N)
            end_time = time.perf_counter()
            print(f"KS.percentiles:  {runtime:.2f} seconds")

        runtime = end_time - start_time

        start_time = time.perf_counter()
        # fig = plot_data(data, data_to_compare, v10, v50, v90, frequency, dx)
        # c2.plotly_chart(fig)
        fig2 = plot_data2(data, data_to_compare, v10, v50, v90, x_values, frequency, dx)
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print(f"Plot data 2:  {runtime:.2f} seconds")

        c12.pyplot(fig2)
        print("-----------")
        # ci = KS.confidence_intervall(data)
        # st.info(ci)
