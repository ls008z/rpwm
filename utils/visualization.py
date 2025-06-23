# plotly based visualization functions

import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np


def plot_3d_scatter(
    df,
    x_col,
    y_col,
    z_col,
    color_col=None,
    size_col=None,
    fit_plane=False,
    title="3D Scatter Plot",
):
    """
    Plots a 3D scatter plot using Plotly.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        x_col (str): The column name for the x-axis.
        y_col (str): The column name for the y-axis.
        z_col (str): The column name for the z-axis.
        color_col (str, optional): The column name for the color dimension.
        size_col (str, optional): The column name for the size dimension.
        title (str): The title of the plot.

    Returns:
        None
    """
    marker_dict = {"opacity": 0.8, "colorscale": "Viridis"}
    if size_col is not None:
        marker_dict["size"] = df[size_col]
    else:
        marker_dict["size"] = 2  # default marker size

    if color_col is not None:
        marker_dict["color"] = df[color_col]
    else:
        marker_dict["color"] = "blue"  # default marker color

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=df[x_col],
                y=df[y_col],
                z=df[z_col],
                mode="markers",
                marker=marker_dict,
            )
        ]
    )
    if fit_plane:
        # Fit a plane to the data
        xx = df[[x_col, y_col]].values
        y = df[z_col].values
        model = LinearRegression()
        model.fit(xx, y)
        x_range = np.linspace(df[x_col].min(), df[x_col].max(), 50)
        y_range = np.linspace(df[y_col].min(), df[y_col].max(), 50)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        z_grid = model.predict(np.c_[x_grid.ravel(), y_grid.ravel()]).reshape(
            x_grid.shape
        )
        fig.add_surface(
            x=x_grid,
            y=y_grid,
            z=z_grid,
            colorscale="Viridis",
            showscale=False,
            opacity=0.5,
        )
    # Set the layout of the plot
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    # Show the plot
    fig.show()
