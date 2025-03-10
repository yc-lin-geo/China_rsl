import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe

### This file contains some additional functions that are used for generate figures in Lin et al., China sea-level budget paper ###
### The major spatiotemporal hierarchical model is defined by scripts in PSTHM folder (PlaeoSTeHM) ###

def gp_kernel_decomposition(gpr, pred_matrix, global_kernel, global_kernel_fast, 
                            local_nl_kernel, local_nl_kernel_fast, 
                            regional_nl_kernel, regional_nl_kernel_fast, 
                            regional_linear_kernel):
    """
    Decompose the Gaussian process kernel into global, local non-linear, regional non-linear, and regional linear components.

    Args:

    - gpr: A trained Gaussian process regression model.
    - pred_matrix: A matrix of prediction points.
    - global_kernel: The global kernel function.
    - global_kernel_fast: The fast global kernel function.
    - local_nl_kernel: The local non-linear kernel function.
    - local_nl_kernel_fast: The fast local non-linear kernel function.
    - regional_nl_kernel: The regional non-linear kernel function.
    - regional_nl_kernel_fast: The fast regional non-linear kernel function.
    - regional_linear_kernel: The regional linear kernel function.
    
    Returns:
    - A tuple of tuples containing the mean and variance for each component:

    ((global_mean, global_var), (local_nl_mean, local_nl_var),
    (regional_nl_mean, regional_nl_var), (regional_l_mean, regional_l_var))
    """

    # Calculate the inverse of K + noise term
    test_K = torch.inverse(gpr.kernel(gpr.X) + gpr.noise)
    
    # Compute combined kernels for K_star
    combined_global_K_star = (global_kernel(gpr.X, pred_matrix) + global_kernel_fast(gpr.X, pred_matrix)).T
    combined_local_nl_K_star = (local_nl_kernel(gpr.X, pred_matrix) + local_nl_kernel_fast(gpr.X, pred_matrix)).T
    combined_regional_nl_K_star = (regional_nl_kernel(gpr.X, pred_matrix) + regional_nl_kernel_fast(gpr.X, pred_matrix)).T
    combined_regional_l_K_star = (regional_linear_kernel(gpr.X, pred_matrix)).T

    # Compute combined kernels for K_star_star
    combined_global_K_star_star = (global_kernel(pred_matrix) + global_kernel_fast(pred_matrix))
    combined_local_nl_K_star_star = (local_nl_kernel(pred_matrix) + local_nl_kernel_fast(pred_matrix))
    combined_regional_nl_K_star_star = (regional_nl_kernel(pred_matrix) + regional_nl_kernel_fast(pred_matrix))
    combined_regional_l_K_star_star = (regional_linear_kernel(pred_matrix))

    # Calculate means
    global_mean = (combined_global_K_star @ test_K @ gpr.y)
    local_nl_mean = (combined_local_nl_K_star @ test_K @ gpr.y)
    regional_nl_mean = (combined_regional_nl_K_star @ test_K @ gpr.y)
    regional_l_mean = (combined_regional_l_K_star @ test_K @ gpr.y)

    # Calculate variances
    global_var = combined_global_K_star_star - combined_global_K_star @ test_K @ combined_global_K_star.T
    local_nl_var = combined_local_nl_K_star_star - combined_local_nl_K_star @ test_K @ combined_local_nl_K_star.T
    regional_nl_var = combined_regional_nl_K_star_star - combined_regional_nl_K_star @ test_K @ combined_regional_nl_K_star.T
    regional_l_var = combined_regional_l_K_star_star - combined_regional_l_K_star @ test_K @ combined_regional_l_K_star.T


    return ((global_mean, global_var), (local_nl_mean, local_nl_var), 
            (regional_nl_mean, regional_nl_var), (regional_l_mean, regional_l_var))

def calculate_sdsl(cdu, q, g, W_m, H_m, interp_x):
    """
    Calculate the sterodynamic sea level (\langle \zeta \rangle) along the estuary.
    
    Parameters:
    - cdu: Product of the drag coefficient and reference velocity scale.
    - q: Streamflow at the origin in m^3/s.
    - g: Acceleration due to gravity in m/s^2.
    - W_m: Width profile of the estuary in meters (NumPy array).
    - H_m: Depth profile of the estuary in meters (NumPy array).
    - interp_x: Interpolated distances along the river in kilometers (NumPy array).
    
    Returns:
    - zeta_values: Calculated ocean dynamic sea level (\langle \zeta \rangle) as a NumPy array.
    """
    # Initialize an array to store the calculated zeta values
    zeta_values = []
    
    # Perform the integration for each segment starting from each x in interp_x
    for i in range(len(interp_x)):
        # Calculate the integrand from the current index to the end
        integrand = 1 / (H_m[i:]**2 * W_m[i:])
        if i < len(interp_x) - 1:  # Ensure there's more than one point for dx calculation
            dx_m = np.diff(interp_x[i:]) * 1000  # Convert dx from km to m for integration
            # Perform numerical integration using the trapezoidal rule for the current segment
            integral_result = np.trapz(integrand, dx=dx_m)
        else:
            integral_result = 0  # No integration possible with a single point
        
        # Calculate ocean dynamic sea level (zeta) in meters for the current segment
        zeta = (cdu * q / g) * integral_result
        zeta_values.append(zeta)
    
    return np.array(zeta_values)



def load_and_plot_budget(site_name):
    """
    Load data from a china cith budget file and create a plot with prediction bounds in the main plot
    and stacked bars in the inset plot.
    
    Args:
        site_name (str): Name of the site (e.g., 'Fuzhou Fujian')
    """
    try:
        # Load the saved NetCDF file
        ds = xr.open_dataset(f'../Data/data_{site_name.replace(" ", "_")}.nc')
        signal_list_inset = [
            ds.mid_global_mean.values,
            ds.mid_GRD_mean.values,
            ds.mid_regional_l_mean.values,
            ds.mid_regional_nl_mean.values,
            ds.mid_local_nl_mean.values
        ]
        colors = ['C0', 'C4', 'C1', 'C3', 'C2']  # Colors matching the original plot
        # Create the main figure
        fig = plt.figure(figsize=(12, 4))
        ax = plt.subplot(111)
        
        # Plot the prediction mean and uncertainty bounds (high resolution, 50-year steps)
        pe = [mpe.withStroke(linewidth=4.5, foreground="w")]
        ax.plot(ds.time_high.values, ds.high_pred_mean.values, color='k', linewidth=3, 
                path_effects=pe, linestyle='-')
        
        # Update path effects for the uncertainty bounds
        pe = [mpe.withStroke(linewidth=5, foreground="w")]
        ax.plot(ds.time_high.values, ds.high_pred_mean.values + ds.high_pred_std.values, 
                color='k', linewidth=2, linestyle='--', dashes=(5, 5), dash_capstyle='round')
        ax.plot(ds.time_high.values, ds.high_pred_mean.values - ds.high_pred_std.values, 
                color='k', linewidth=2, linestyle='--', dashes=(5, 5), dash_capstyle='round')

        # Set limits and labels for the main plot
        ax.set_xlim(11125, 0)
        ax.set_ylim(-55, 20)
        ax.set_xlabel('Age (BP)', fontsize=20)
        ax.set_ylabel('RSL (m)', fontsize=20)
        ax.text(6000, 12, site_name, weight='extra bold', fontsize=25, ha="center")
        for i in range(len(ds.time_mid)):
            time_index = i
            upper_bound = 0
            lower_bound = 0
            for i2 in range(5):
                test_signal = signal_list_inset[i2][time_index]
                if test_signal < 0:
                    ax.add_patch(plt.Rectangle((ds.time_mid.values[time_index] - 125, lower_bound), 
                                                2 * 125, -abs(test_signal), 
                                                fill=True, fc=colors[i2], ec='k', linewidth=0.5, alpha=0.7))
                else:
                    ax.add_patch(plt.Rectangle((ds.time_mid.values[time_index] - 125, upper_bound), 
                                                2 * 125, abs(test_signal), 
                                                fill=True, fc=colors[i2], ec='k', linewidth=0.5, alpha=0.7))
                if test_signal > 0:
                    upper_bound += test_signal
                else:
                    lower_bound += test_signal

        # Add inset plot
        left, bottom, width, height = 0.482, 0.110, 0.418, 0.45
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.tick_params(labelbottom=False)  # No x-axis labels for inset
        ax2.tick_params(labelleft=True)
        ax2.set_xticks([])  # No x-axis ticks
        ax2.set_yticks([-4, 0, 4])  # Y-axis ticks matching the original plot
        ax2.tick_params(axis='y')

        # Stacked bars for inset (low resolution, 100-year steps)
        signal_list_inset = [
            ds.low_global_mean.values,
            ds.low_GRD_mean.values,
            ds.low_regional_l_mean.values,
            ds.low_regional_nl_mean.values,
            ds.low_local_nl_mean.values
        ]
        

        # Create stacked bars using add_patch
        for i in range(len(ds.time_low)):
            time_index = i
            upper_bound = 0
            lower_bound = 0
            for i2 in range(5):
                test_signal = signal_list_inset[i2][time_index]
                if test_signal < 0:
                    ax2.add_patch(plt.Rectangle((ds.time_low.values[time_index] - 50, lower_bound), 
                                                2 * 50, -abs(test_signal), 
                                                fill=True, fc=colors[i2], ec='k', linewidth=0.5, alpha=0.7))
                else:
                    ax2.add_patch(plt.Rectangle((ds.time_low.values[time_index] - 50, upper_bound), 
                                                2 * 50, abs(test_signal), 
                                                fill=True, fc=colors[i2], ec='k', linewidth=0.5, alpha=0.7))
                if test_signal > 0:
                    upper_bound += test_signal
                else:
                    lower_bound += test_signal

        # Plot prediction mean and uncertainty for inset
        ax2.plot(ds.time_high.values, ds.high_pred_mean.values, color='k', linewidth=3, 
                 path_effects=pe, linestyle='-')
        ax2.plot(ds.time_high.values, ds.high_pred_mean.values + ds.high_pred_std.values, 
                 color='k', linewidth=2, linestyle='--', dashes=(6, 6), dash_capstyle='round')
        ax2.plot(ds.time_high.values, ds.high_pred_mean.values - ds.high_pred_std.values, 
                 color='k', linewidth=2, linestyle='--', dashes=(6, 6), dash_capstyle='round')

        # Set limits for the inset
        ax2.set_xlim(6000, 0)
        ax2.set_ylim(-5.5, 5.5)

        # Display the plot
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: The file 'data_{site_name.replace(' ', '_')}.nc' was not found. Please ensure it has been generated.")
    except KeyError as e:
        print(f"Error: Missing variable {e} in the NetCDF file. Check if the data was saved correctly.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")