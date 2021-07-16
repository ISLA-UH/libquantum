"""
This module contains 2D and 3D geolocation scatter plots
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List


def location_2d(x: Union[np.ndarray, float],
                y: Union[np.ndarray, float],
                color_guide: Union[np.array, List],
                fig_title: str,
                x_label: str,
                y_label: str,
                color_label: str,
                dot_size: float = 16.,
                color_map: str = "inferno") -> None:
    """
    Geolocation in 2D plane (plane slice)

    :param x: array-like x data positions
    :param y: array-like y data positions
    :param color_guide: array-like or list of colors or color
    :param fig_title: title of figure
    :param x_label: string for x-axis label
    :param y_label: string for y-axis label
    :param color_label: set a label that will be displayed in the legend
    :param dot_size: the marker size in points**2. Default is 16.0
    :param color_map: a Matplotlib Colormap instance or registered colormap name. Default is "inferno"
    :return: plot
    """
    fig, ax = plt.subplots()
    im = ax.scatter(x, y, s=dot_size, c=color_guide, marker='o', cmap=color_map)
    plt.title(fig_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax_colorbar = fig.colorbar(im, ax=ax)
    ax_colorbar.set_label(color_label)
    fig.tight_layout()


def location_3d(x: Union[np.ndarray, float],
                y: Union[np.ndarray, float],
                z: Union[np.ndarray, float],
                color_guide: Union[np.array, List],
                fig_title: str,
                x_label: str,
                y_label: str,
                z_label: str,
                color_label: str,
                dot_size: int = 16,
                color_map: str = "inferno",
                azimuth_degrees: float = -115,
                elevation_degrees: float = 34,
                plot_line: bool = False) -> None:
    """
    Geolocation in 3D

    :param x: array-like x data positions, e.g., longitude
    :param y: array-like y data positions, e.g., latitude
    :param z: array-like z data positions, e.g., altitude
    :param color_guide: array-like or list of colors or color
    :param fig_title: title of figure
    :param x_label: string for x-axis label
    :param y_label: string for y-axis label
    :param z_label: string for z-axis label
    :param color_label: set a label that will be displayed in the legend
    :param dot_size: the marker size in points**2. Default is 16.0
    :param color_map: a Matplotlib Colormap instance or registered colormap name. Default is "inferno"
    :param azimuth_degrees: azimuth to view plot in degrees. Default is -115
    :param elevation_degrees: elevation to view plot in degrees. Default is 34
    :param plot_line: default is False
    :return: plot
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if plot_line:
        ax.plot(x, y, z, 'gray')
    im = ax.scatter(x, y, z, s=dot_size, c=color_guide, marker='o', cmap=color_map)
    ax.view_init(azim=azimuth_degrees, elev=elevation_degrees)
    plt.title(fig_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax_colorbar = fig.colorbar(im, ax=ax)
    ax_colorbar.set_label(color_label)
    fig.tight_layout()


def loc_quiver_3d(x: Union[np.ndarray, float],
                  y: Union[np.ndarray, float],
                  z: Union[np.ndarray, float],
                  u: Union[np.ndarray, float],
                  v: Union[np.ndarray, float],
                  w: Union[np.ndarray, float],
                  color_guide: Union[np.array, List],
                  fig_title: str,
                  x_label: str,
                  y_label: str,
                  z_label: str,
                  color_label: str,
                  dot_size: float = 16.,
                  color_map: str = "inferno",
                  azimuth_degrees: float = -115,
                  elevation_degrees: float = 34,
                  arrow_length: float = 0.5) -> None:
    """
    3D speed quiver plot

    :param x: array-like x data positions, e.g., longitude
    :param y: array-like y data positions, e.g., latitude
    :param z: array-like z data positions, e.g., altitude
    :param u: array-like x speed component in meters per second
    :param v: array-like y speed component in meters per second
    :param w: array-like z speed component in meters per second
    :param color_guide: array-like or list of colors or color
    :param fig_title: title of figure
    :param x_label: string for x-axis label
    :param y_label: string for y-axis label
    :param z_label: string for z-axis label
    :param color_label: set a label that will be displayed in the legend
    :param dot_size: the marker size in points**2. Default is 16.0
    :param color_map: a Matplotlib Colormap instance or registered colormap name. Default is "inferno"
    :param azimuth_degrees: azimuth to view plot in degrees. Default is -115
    :param elevation_degrees: elevation to view plot in degrees. Default is 34
    :param arrow_length: length of arrow. Default is 0.5
    :return: plot
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    im = ax.scatter(x, y, z, s=dot_size, c=color_guide, marker='o', cmap=color_map)
    ax.quiver(x, y, z, u, v, w, length=arrow_length)
    ax.view_init(azim=azimuth_degrees, elev=elevation_degrees)
    plt.title(fig_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax_colorbar = fig.colorbar(im, ax=ax)
    ax_colorbar.set_label(color_label)
    fig.tight_layout()


def loc_overlay_3d(x1: Union[np.ndarray, float],
                   y1: Union[np.ndarray, float],
                   z1: Union[np.ndarray, float],
                   color1: str,
                   legend1: str,
                   x2: Union[np.ndarray, float],
                   y2: Union[np.ndarray, float],
                   z2: Union[np.ndarray, float],
                   color2: str,
                   legend2: str,
                   fig_title: str,
                   x_label: str,
                   y_label: str,
                   z_label: str,
                   dot_size1: float = 10.,
                   dot_size2: float = 8.,
                   alpha1: float = 1.,
                   alpha2: float = 0.6,
                   azimuth_degrees: float = -115,
                   elevation_degrees: float = 34) -> None:
    """
    Overlay two geolocations in 3D

    :param x1: array-like x data positions for geolocation 1, e.g., longitude
    :param y1: array-like y data positions for geolocation 1, e.g., latitude
    :param z1: array-like z data positions for geolocation 1, e.g., altitude
    :param color1: color of the line for geolocation 1
    :param legend1: label of the line for geolocation 1
    :param x2: array-like x data positions for geolocation 2, e.g., longitude
    :param y2: array-like y data positions for geolocation 2, e.g., latitude
    :param z2: array-like z data positions for geolocation 2, e.g., altitude
    :param color2: color of the line for geolocation 2
    :param legend2:  label of the line for geolocation 2
    :param fig_title: title of figure
    :param x_label: string for x-axis label
    :param y_label: string for y-axis label
    :param z_label: string for z-axis label
    :param dot_size1: the marker size in points**2 for geolocation 1. Default is 10.0
    :param dot_size2: the marker size in points**2 for geolocation 2. Default is 8.0
    :param alpha1: alpha value used for blending for geolocation 1. Default is 1.0
    :param alpha2: alpha value used for blending for geolocation 2. Default is 0.6
    :param azimuth_degrees: azimuth to view plot in degrees. Default is -115
    :param elevation_degrees: elevation to view plot in degrees. Default is 34
    :return: plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(x1, y1, z1, markersize=dot_size1, color=color1, marker='o', label=legend1, alpha=alpha1)
    ax.plot(x2, y2, z2, markersize=dot_size2, color=color2, marker='o', label=legend2, alpha=alpha2)
    ax.view_init(azim=azimuth_degrees, elev=elevation_degrees)
    plt.title(fig_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.legend()
    fig.tight_layout()
