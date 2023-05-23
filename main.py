import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Base directory where all directories are located
BASE_DIR = "/Users/chraibi/sciebo/Rimea_VV/single_file/2ColData"


def load_directories(base_dir):
    """Function to load all directories within the base directory"""
    return [
        dir
        for dir in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, dir))
    ]


def load_data_from_dir(directory):
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


def plot_data(data, freq):
    """Function to plot data"""
    fig, ax = plt.subplots()
    for dir, df in data.items():
        ax.plot(df["rho"][::freq], df["velocity"][::freq], ".", label=dir)
    plt.legend()
    plt.xlabel(r"$\rho\;\; / 1/m$")
    plt.ylabel(r"$v\;\; / m/s$")
    plt.xlim([0, 8])
    plt.ylim([-0.5, 3])

    return fig


st.title("Directory Data Visualization")

# load directories
directories = load_directories(BASE_DIR)
directories.sort()
# selected_directories = st.multiselect("Select directories", directories)
frequency = st.number_input(
    "Enter the frequency of the points to be plotted", min_value=1, value=10
)
c1, _, c2 = st.columns((0.2, 0.5, 0.7))
col1, col2, col3 = st.columns(3)
selected_directories = [dir for dir in directories if c1.checkbox(f"{dir}")]


data = {}
for directory in selected_directories:
    data[directory] = load_data_from_dir(os.path.join(BASE_DIR, directory))

fig = plot_data(data, frequency)
c2.pyplot(fig)
