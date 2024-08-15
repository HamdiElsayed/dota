import numpy as np

def flux_projection(
        beamlet_entrance: np.ndarray,
        beamlet_direction: np.ndarray,
        sigmas_xy: np.ndarray,
        angular_divergence: np.ndarray,
        shape: list) -> np.ndarray:
   
    """Generate  flux projection, directed in the given angle and with the given positional spread and angular divergence.

    Args:
        beamlet_entrance (np.ndarray): Beamlet entrance coordinate (x, y, z) [mm]
        beamlet_direction (np.ndarray): Beamlet direction angles: (theta_x, theta_y) [deg]
        sigmas_xy (np.ndarray): Initial spatial spread of the proton flux (sigma_x, sigma_y) [mm]
        angular_divergence (np.ndarray): Angular divergence in mrad (div_x, div_y) [mrad]
        shape (list): Shape of the output array.

    Returns:
        np.ndarray: Proton flux projection.
    """
    # Auxiliary functions:
    R_x = lambda theta: np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    R_y = lambda theta: np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

    # Main part:
    x_0, y_0, z_0 = beamlet_entrance
    x = np.arange(0, shape[1], 1)
    y = np.arange(0, shape[0], 1)
    z = np.arange(0, shape[2], 1)
    [xx, yy, zz] = np.meshgrid(x, y, z)

    theta_x_deg, theta_y_deg = beamlet_direction
    theta_x = np.deg2rad(theta_x_deg)
    theta_y = np.deg2rad(theta_y_deg)

    [x_t, y_t, z_t] = R_y(theta_x) @ R_x(theta_y) @ np.array([
        xx.flatten() - x_0, 
        yy.flatten() - y_0, 
        zz.flatten() - z_0
    ])
    x_t = x_t.reshape(xx.shape)
    y_t = y_t.reshape(yy.shape)
    z_t = z_t.reshape(zz.shape)

    # Convert angular divergence from mrad to rad
    div_x, div_y = np.deg2rad(angular_divergence / 1000)

    # Calculate the total spread at depth z_t
    sigma_x_z = np.sqrt(sigmas_xy[0]**2 + (np.tan(div_x) * z_t)**2)
    sigma_y_z = np.sqrt(sigmas_xy[1]**2 + (np.tan(div_y) * z_t)**2)

    # Compute flux
    coef = 1 / (2 * np.pi * sigma_x_z * sigma_y_z)
    flux = coef * np.exp(-x_t**2 / (2 * sigma_x_z**2) - y_t**2 / (2 * sigma_y_z**2))

    return flux