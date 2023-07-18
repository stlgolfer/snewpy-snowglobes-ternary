import snewpyternary as t
import snowglobes_wrapper
from model_wrappers import snewpy_models, sn_model_default_time_step
import data_handlers
from meta_analysis import process_flux, process_detector
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u

def estimate_cxn(config: t.MetaAnalysisConfig, flux_chan: int, cxn_truth_fname: str, cxn_truth_chan_key: str):
    time_bins_x_axis, dt_not_needed = snowglobes_wrapper.calculate_time_bins(
        config.model_file_paths[0],
        config.model_type,
        deltat=sn_model_default_time_step(config.model_type),
        log_bins=True,
        presn=False
    )

    times_unitless = [m.value for m in time_bins_x_axis]
    temp1 = np.array([times_unitless[0] + 0.5])
    temp2 = times_unitless[1:]
    temp3 = times_unitless[0:-1]
    temp4 = np.subtract(np.array(temp2), np.array(temp3))
    dts = np.concatenate((np.array(temp1), temp4))

    flux_scatter_data, raw_data, labeled = process_flux(config, 0)
    # in a time bin, the truth flux is binned in GeV
    labeled_transposed = np.transpose(labeled)
    # plot_data, raw_data_det, l_data, zeta, sigma_average = process_detector(config, 0, det_name)

    # region let's just try to plot phi_t over time, flux vs time, and flux vs energy
    phi_t_fig, phi_t_ax = plt.subplots(1, 1)

    phi_t_over_time = np.zeros_like(times_unitless)
    flux_vs_time = np.zeros_like(times_unitless)
    for l in range(len(labeled)):
        phi_t_over_time[l] = np.sum(labeled[l][flux_chan]) * 0.2e-3  # * dts[l]
        flux_vs_time[l] = np.sum(labeled[l][flux_chan]) * 0.2e-3
    phi_t_ax.scatter(times_unitless, phi_t_over_time)
    phi_t_ax.set_title(r'$\phi_t$ for Nakazato 0 Fluence')
    phi_t_ax.set_xlabel("Time (s)")
    phi_t_ax.set_ylabel(r'$neutrinos/cm^2$')
    phi_t_ax.set_xscale('log')
    phi_t_fig.tight_layout()
    phi_t_fig.show()

    flux_vs_time_fig, flux_vs_time_ax = plt.subplots(1, 1)
    flux_vs_time_ax.bar(times_unitless, flux_vs_time, width=dts)
    flux_vs_time_ax.set_title("Flux vs time, integrated over energy")
    flux_vs_time_ax.set_xlabel("Time Mid-Point (s)")
    flux_vs_time_ax.set_ylabel(r'$\frac{neutrinos}{{cm^2}*s}$')
    flux_vs_time_fig.tight_layout()
    flux_vs_time_AUC = np.sum(np.multiply(flux_vs_time, dts))
    print(f"AUC for Flux vs time is {flux_vs_time_AUC}")
    flux_vs_time_fig.show()

    flux_vs_energy = np.zeros_like(labeled[0][0])
    for en in range(len(flux_vs_energy)):
        flux_vs_energy[en] = np.sum(np.multiply(labeled_transposed[en][flux_chan], dts))
    flux_vs_energy_fig, flux_vs_energy_ax = plt.subplots(1, 1)
    flux_vs_energy_ax.bar(labeled[0][0], flux_vs_energy, width=0.2e-3)
    flux_vs_energy_ax.set_title("Flux vs energy, integrated over time")
    flux_vs_energy_ax.set_xlabel('Mid-Point Energy (GeV)')
    flux_vs_energy_ax.set_ylabel(r'$\frac{neutrinos}{{cm}^2 * GeV}$')
    # then find the AUC
    flux_vs_energy_AUC = np.sum(flux_vs_energy) * 0.2e-3
    print(f"AUC for Flux vs energy is {flux_vs_energy_AUC}")
    print(f'AUC for phi_t is {np.sum(phi_t_over_time)}')
    flux_vs_energy_fig.show()

    print(f"AUC ratio fve/fvt = {flux_vs_energy_AUC / flux_vs_time_AUC}")
    # endregion

    # truth_flux_fig, truth_flux_ax = plt.subplots(1,1)
    # truth_flux_ax.scatter()

    # make a spectrogram of the actual xscn from snowglobes (it will be time-invariant)
    cxn_truth = pd.read_csv(f'./snowglobes_cxns/{cxn_truth_fname}.csv')  # only want the 'nu_e_bar' channel
    # now we'll also need to downsample the cross since the cross section has far more bins (around double) than flux
    # this is also time-invariant so only need to compute once
    print("Loaded truth xscn")

    cxn_truth_energy = 10 ** np.array(cxn_truth['energy'])

    truth_calculation = np.multiply(
        cxn_truth_energy,
        np.array(cxn_truth[cxn_truth_chan_key])
    )

    truth_calculation_no_downsample = np.copy(truth_calculation) * 1e-38

    truth_calculation = np.nan_to_num(truth_calculation, nan=0.0)

    # downsample the truth flux with 1e38 cm^2 scale since python might have a hard time interpolating such small values
    truth_calculation = np.interp(labeled[0][0], cxn_truth_energy, truth_calculation) * 1e-38
    print("Truth CXN Downsampled")

    # region plot nu_e_bar_cxn_truth
    cxn_truth_fig, cxn_truth_ax = plt.subplots(1, 1)
    cxn_truth_ax.scatter(cxn_truth_energy, truth_calculation_no_downsample, label="CXN No Downsample")
    cxn_truth_ax.scatter(labeled[0][0], truth_calculation, label="CXN Downsampled", alpha=0.4)
    cxn_truth_ax.set_xlabel('Energy (GeV)')
    cxn_truth_ax.set_ylabel(r'${cm}^2$')
    cxn_truth_ax.set_title(r'$\sigma$ Truth')
    cxn_truth_ax.set_xlim(1e-3, 1e-1)
    cxn_truth_ax.set_ylim(1e-44, 1e-39)
    cxn_truth_ax.set_xscale('log')
    cxn_truth_ax.set_yscale('log')
    cxn_truth_fig.legend()
    cxn_truth_fig.show()
    # endregion

    # region plot truth flux average energy over time
    truth_flux_average_energy = np.zeros_like(times_unitless)
    for lt in range(len(labeled)):
        # find int E*F(E) dE / int F(E) dE, though in all fairness dE will cancel here
        truth_flux_average_energy[lt] = np.sum(np.multiply(labeled[lt][0], labeled[lt][flux_chan] * dts[lt])) / np.sum(
            labeled[lt][flux_chan] * dts[lt])

    tf_average_energy_fig, tf_average_energy_ax = plt.subplots(1, 1, figsize=(8, 8))
    tf_average_energy_ax.scatter(times_unitless, truth_flux_average_energy)
    tf_average_energy_ax.set_xscale('log')
    tf_average_energy_ax.set_xlabel('Time (s)')
    tf_average_energy_ax.set_ylabel('Flux-Weighted Average Energy (GeV)')
    tf_average_energy_ax.set_title(r'$\mathbb{E}(E_\nu|F(E_\nu))$')
    tf_average_energy_fig.show()
    # endregion

    # we want to make the flux average cross-section by using the cross-section from snowglobes
    # go through each time bin

    sigma_average = np.zeros_like(times_unitless)
    for time in range(len(times_unitless)):
        sigma_average[time] = np.sum(np.multiply(0.2e-3 * labeled[time][flux_chan], truth_calculation)) / phi_t_over_time[time]

    # now plot the flux-averaged cxn
    fig, ax = plt.subplots(1, 1)
    ax.scatter(times_unitless, sigma_average)
    ax.set_ylabel(r'$<\sigma>$')
    ax.set_xlabel('Time (s)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(rf'$<\sigma>$ for Nakazato 0 IBD')
    fig.savefig('./sigma for nakazato 0 ibd.png')
    fig.show()

if __name__ == '__main__':
    config = t.MetaAnalysisConfig(
        snewpy_models['Nakazato_2013'],
        [0],
        'AdiabaticMSW_IMO',
        proxy_config=data_handlers.ConfigBestChannel()
    )
    estimate_cxn(config, 1)
