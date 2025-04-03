from observed_data_retriever import ObservedDataRetriever, plot_obs_reservoir_dynamics


if __name__ == "__main__":
    retriever = ObservedDataRetriever()

    # Define gauges and mappings
    inflow_gauges = {
        "01449360": "beltzvilleCombined",
        "01447500": "fewalter",
        "01428750": "prompton"
    }
    release_gauges = {
        "01449800": "beltzvilleCombined",
        "01447800": "fewalter",
        "01429000": "prompton",
        "01470960": "blueMarsh"
    }
    storage_gauges = {
        "01449790": "beltzvilleCombined",
        "01447780": "fewalter",
        "01428900": "prompton",
        "01470870": "blueMarsh"
    }

    storage_curves = {
        "beltzvilleCombined": "raw/beltzvilleCombined_storage_curve.csv",
        "fewalter": "raw/fewalter_storage_curve.csv",
        "prompton": "raw/prompton_storage_curve.csv",
        "blueMarsh": "raw/blueMarsh_storage_curve.csv"
    }

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
