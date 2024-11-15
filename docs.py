"""
References to the data
"""

import streamlit as st


def methods():
    st.header("Methods")

    st.markdown(
        r"""
        :blue[**Perentiles**]  
        $k-th$ percentile, is a score below which a given percentage $k$ of scores in its frequency distribution falls
        Here,  $k \in [10, 50, 90]$.

        The density axis is devided in $dx$ intervals. Then the number of data points within an interval of length $dx$ are counted.  
        $N$ defines the mimnimum number of necessary data within each interval.    
        """
    )
    st.markdown(
        """
        :blue[**KS-Test**]  
        Kolmogorov-Smirnov test according to this paper:

        """
    )
    st.info(
        r"""
        **Automated Quality Assessment of Space-Continuous Models for Pedestrian Dynamics**  
        [10.1007/978-3-030-11440-4\_35](https://link.springer.com/content/pdf/10.1007/978-3-030-11440-4\_35.pdf)  
        2019
              """
    )
    st.markdown(
        """              
        A KS-score equal to zero, means the two datasets that are being compared, are perfectly matching each other.
        
    """
    )


def references():
    st.header("Data")
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
    c1.markdown(
        """
                - :blue[**27_new_beginnings**]  
                Wheelchair and Phone use During Single File Pedestrian Movement

        DOI: [10.1007/978-981-99-7976-9_23](https://link.springer.com/chapter/10.1007/978-981-99-7976-9_23)  
    2022
    """
    )
