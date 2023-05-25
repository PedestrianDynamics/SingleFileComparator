"""
References to the data
"""
import streamlit as st


def references():
    c1, c2 = st.columns(2)
    c1.markdown(
        """
        - :blue[**01_FD_germany_seyfried2005**]  
        The fundamental diagram of pedestrian movement revisited,  
        DOI: [10.1088/1742-5468/2005/10/P10002](https://iopscience.iop.org/article/10.1088/1742-5468/2005/10/P10002),  
        2005
        """
    )
    c2.markdown(
        """
    - :blue[**02_culture_india_chattaraj2013**]  
    Comparison of pedestrian fundamental diagram across cultures
    DOI: [10.1142/S0219525909002209](https://www.worldscientific.com/doi/abs/10.1142/S0219525909002209)  
    2009
    """
    )
