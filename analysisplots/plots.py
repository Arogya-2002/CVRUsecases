import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Create a title for the Streamlit app
st.title("Plot Display on Button Click")

# Create a button and store its state
if st.button('Show Plot'):
    # If button is clicked, generate and display the plot
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    
    # Display the plot
    st.pyplot(fig)
