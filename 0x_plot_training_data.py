import matplotlib.pyplot as plt

from methods.load.observations import get_observational_training_data

from methods.config import DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR
from methods.config import reservoir_options, policy_type_options


for reservoir in reservoir_options:
    
    # load data
    inflow_obs, release_obs, storage_obs = get_observational_training_data(
        reservoir_name=reservoir,
        data_dir = PROCESSED_DATA_DIR,
        as_numpy=False,
        scaled_inflows=True
    )
    
    # Plot 3-panel inflow, storage release
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig.suptitle(f"Reservoir: {reservoir}", fontsize=16)
    axs[0].plot(inflow_obs.index, inflow_obs.values, label='Inflow', color='blue')
    axs[0].set_ylabel('Inflow')
    axs[1].plot(storage_obs.index, storage_obs.values, label='Storage', color='green')
    axs[1].set_ylabel('Storage')
    axs[2].plot(release_obs.index, release_obs.values, label='Release', color='red')
    axs[2].set_ylabel('Release')
    axs[2].set_xlabel('Date')
    axs[2].legend()
    plt.savefig(f"./training_data_{reservoir}_inflow_storage_release.png")