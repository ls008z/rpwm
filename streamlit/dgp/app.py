import streamlit as st
import matplotlib.pyplot as plt
import sys
import os
import dgp.diagonal as diagonal
import utils.visualization as viz

st.title("Diagonal DGP Demo")

# Parameters
n = st.sidebar.slider("Samples", 100, 2000, 1000, step=100)
error_var = st.sidebar.slider("Error Variance", 0.1, 3.0, 1.0, step=0.1)

if st.button("Generate"):
    # Generate data
    df = diagonal.generate_data(n, error_var)

    st.subheader("Data Summary")
    st.dataframe(
        df[
            ["X1", "X2", "X3", "X4", "D", "Y", "Y(0)", "Y(1)", "p_score", "effect"]
        ].describe()
    )

    st.metric("ATE", f"{df['effect'].mean():.3f}")

    col1, col2 = st.columns(2)

    with col1:
        # Effect scatter
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(
            df["X1"], df["X2"], c=df["effect"], cmap="RdBu_r", alpha=0.6
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="X2=X1")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_title("Treatment Effect (X2-X1)")
        plt.colorbar(scatter)
        ax.legend()
        st.pyplot(fig)

    with col2:
        # Propensity scores
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(df["p_score"], bins=20, alpha=0.7)
        ax.set_xlabel("Propensity Score")
        ax.set_title("Treatment Assignment Probability")
        st.pyplot(fig)

        # Outcomes by treatment
        fig, ax = plt.subplots(figsize=(6, 3))
        df[df["D"] == 0]["Y"].hist(alpha=0.5, label="Control", bins=20)
        df[df["D"] == 1]["Y"].hist(alpha=0.5, label="Treated", bins=20)
        ax.legend()
        ax.set_xlabel("Outcome Y")
        ax.set_title("Observed Outcomes")
        st.pyplot(fig)

    # 3D Plots
    st.subheader("3D Visualizations")

    col3, col4 = st.columns(2)

    with col3:
        # 3D Effect plot
        fig = viz.plot_3d_scatter(
            df,
            x_col="X1",
            y_col="X2",
            z_col="effect",
            color_col="D",
            title="Treatment Effect (X2 - X1)",
            fit_plane=True,
        )
        st.plotly_chart(fig)

    with col4:
        # 3D Propensity plot
        fig = viz.plot_3d_scatter(
            df,
            x_col="X1",
            y_col="X2",
            z_col="p_score",
            color_col="D",
            title="Propensity Score",
            fit_plane=True,
        )
        st.plotly_chart(fig)
