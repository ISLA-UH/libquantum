import matplotlib.pyplot as plt


def location_2d(x, y, color_guide,
                fig_title: str, x_label: str, y_label: str, color_label: str,
                dot_size: int = 16, color_map: str = "inferno"):
    """
    Geolocation in 2D plane (plane slice)
    :param x:
    :param y:
    :param color_guide:
    :param fig_title:
    :param x_label:
    :param y_label:
    :param color_label:
    :param dot_size:
    :param color_map:
    :return:
    """
    fig, ax = plt.subplots()
    im = ax.scatter(x, y, s=dot_size, c=color_guide, marker='o', cmap=color_map)
    plt.title(fig_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax_colorbar = fig.colorbar(im, ax=ax)
    ax_colorbar.set_label(color_label)
    fig.tight_layout()


def location_3d(x, y, z, color_guide,
                fig_title: str, x_label: str, y_label: str, z_label: str, color_label: str,
                dot_size: int = 16, color_map: str = "inferno",
                azimuth_degrees: float = -115, elevation_degrees: float = 34,
                plot_line: bool = False):

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


def loc_quiver_3d(x, y, z, u, v, w, color_guide,
                  fig_title: str, x_label: str, y_label: str, z_label: str, color_label: str,
                  dot_size: int = 16, color_map: str = "inferno",
                  azimuth_degrees: float = -115, elevation_degrees: float = 34,
                  arrow_length: float = 0.5):
    # TODO: Add arrow color
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


def loc_overlay_3d(x1, y1, z1, color1, legend1,
                   x2, y2, z2, color2, legend2,
                   fig_title: str, x_label: str, y_label: str, z_label: str,
                   dot_size1: int = 10,
                   dot_size2: int = 8,
                   alpha1: float = 1.,
                   alpha2: float = 0.6,
                   azimuth_degrees: float = -115, elevation_degrees: float = 34):

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
