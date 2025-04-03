from methods.preprocessing.observed_data_retriever import ObservedDataRetriever
from methods.plotting.plot_obs_dynamics import plot_obs_reservoir_dynamics

# Directories
from methods.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, FIG_DIR
from .gauge_ids import inflow_gauges, release_gauges, storage_gauges, storage_curves

if __name__ == "__main__":
    retriever = ObservedDataRetriever()

    # Get inflows
    inflows = retriever.get(list(inflow_gauges.keys()), param_cd="00060", label_map=inflow_gauges)
    retriever.save_to_csv(inflows, "inflow")

    # Get releases
    releases = retriever.get(list(release_gauges.keys()), param_cd="00060", label_map=release_gauges)
    retriever.save_to_csv(releases, "release")

    # Get elevations
    elevations = retriever.get(list(storage_gauges.keys()), param_cd="00062", label_map=storage_gauges)
    retriever.save_to_csv(elevations, "elevation_raw")

    # Convert to storage
    storages = retriever.convert_elevation_to_storage(elevations, storage_curves)
    retriever.save_to_csv(storages, "storage")

    # Check for missing days
    for name, df in [("inflow", inflows), ("release", releases), ("storage", storages)]:
        missing = retriever.find_missing_dates(df)
        print(f"{name} missing dates: {len(missing)} days")

    for res in sorted(set(inflows.columns).union(storages.columns).union(releases.columns)):
        print(f"\nPlotting reservoir: {res}")
        plot_obs_reservoir_dynamics(
            I=inflows, 
            S=storages, 
            R=releases,
            reservoir_name=res,
            title=f"{res} Reservoir Observed Data",
            timescale='daily',
            log=True
        )
