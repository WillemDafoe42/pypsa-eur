# Willem Master's Thesis

import pypsa
import matplotlib.pyplot as plt
import pandas as pd
import gc
import math
import numpy as np
import pickle as pkl
import glob


def set_parameters_from_optimized(n, n_optim, ratio_wind, ratio_pv, sec_margin):
    '''
    Function to set optimized parameters from optimized network as nominal parameters
    of the operative network.

    CHANGE IN FUNCTION: wind power allocation esp. in the north was VERY weird. Therefore: Setting p_nom of wind generators
    to p_nom and NOT p_nom_opt
    -----
    Param:
    n: Optimized investment network.
    n_optim: Operative network for redispatch simulation.
    '''
    print("\n\n")
    print(ratio_wind)
    print(ratio_pv)

    # set line capacities to optimized
    n.lines['s_nom_extendable'] = False

    # correct incorrect line capacities

    line_typed_i = n.lines.index[n.lines.type != '']
    # cap_ac_type_i = 0.635
    l_shedding = abs(n_optim.lines_t.mu_upper.mean(axis=0).round(2)).nlargest(2).index.to_list()

    n.lines.loc[line_typed_i, "s_max_pu"] = sec_margin
    n.lines.loc[l_shedding, "s_max_pu"] = 1
    n.lines.loc[l_shedding, "num_parallel"] = 1

    # set link capacities to optimized (HVDC links as well as store out/inflow links)
    if not n.links.empty:
        n.links.loc[links_dc_i, 'p_nom_extendable'] = False

    # set extendable generators to optimized and p_nom_extendable to False
    gen_extend_i = n.generators.index[n.generators.p_nom_extendable]

    # Overwrite renewables with un-optimized capacities to approximate real life better
    gen_onwind = n.generators.index[n.generators.carrier == "onwind"]
    gen_offwind = n.generators.index[(n.generators.carrier == "offwind-ac") | (n.generators.carrier == "offwind-dc")]
    gen_solar = n.generators.index[n.generators.carrier == "solar"]
    gen_biomass = n.generators.index[n.generators.carrier == "biomass"]
    gen_coal = n.generators.index[n.generators.carrier == "coal"]

    # correct incorrect wind data:
    n.generators.loc[gen_onwind, 'p_nom'] = n.generators['p_nom'].reindex(gen_onwind, fill_value=0.) * ratio_wind + 0.01
    n.generators.loc[
        (n.generators["carrier"] == "onwind") & (n.generators["p_nom"] < 200), "p_nom"] = n.generators.p_nom + 50
    # display(n.generators.loc[(n.generators["carrier"] == "onwind"), "p_nom"])

    n.generators.loc[gen_offwind, 'p_nom'] = n.generators['p_nom_max'].reindex(gen_offwind, fill_value=0.) * 0.05 + 0.01
    n.generators.loc[gen_solar, 'p_nom'] = n.generators['p_nom'].reindex(gen_solar, fill_value=0.) * ratio_pv + 0.01
    n.generators.loc[gen_extend_i, 'p_nom_extendable'] = False

    # correct biomass data:
    ratio_biomass = 9.058
    n.generators.loc[gen_biomass, 'p_nom'] = n.generators['p_nom'] * ratio_biomass + 0.01

    # correct coal data:
    ratio_coal = 25000 / n.generators.loc[gen_coal].p_nom.sum()
    print(ratio_coal)
    n.generators.loc[gen_coal, 'p_nom'] = n.generators['p_nom'] * ratio_coal + 0.01

    # Remove abgeschaltetes akw
    print(len(n.buses.index))
    if len(n.buses.index) > 120:
        n.generators.loc[(n.generators.carrier=="nuclear")&(n.generators.bus==n.buses[
            (n.buses.x == 8.390808) & (n.buses.y ==53.343173)].index[0]),"p_nom"] = 0.01
    # display(n.generators[n.generators.carrier == "nuclear"])

    # correct nuclear data:
    ratio_nuc = 9500 / n.generators[n.generators.carrier == "nuclear"].p_nom.sum()
    n.generators.loc[n.generators.carrier == "nuclear", 'p_nom'] = n.generators['p_nom'] * ratio_nuc + 0.01

    # set extendable storage unit power to ooptimized
    n.storage_units['p_nom_extendable'] = False
    # start SOC of storage_units equals state of charge at last snapshot of previous day
    n.storage_units["cyclic_state_of_charge"] = True

    #     display(n_optim.generators.groupby("carrier").sum())
    #     display(n.generators.groupby("carrier").sum())

    # Set Marginal costs of generators to TSO data from jacob

    n.generators.loc[n.generators["carrier"] == "OCGT", "marginal_cost"] = 79.56
    n.generators.loc[n.generators["carrier"] == "CCGT", "marginal_cost"] = 79.56  # * 47/58
    n.generators.loc[n.generators["carrier"] == "coal", "marginal_cost"] = 43
    n.generators.loc[n.generators["carrier"] == "lignite", "marginal_cost"] = 26.98
    n.generators.loc[n.generators["carrier"] == "nuclear", "marginal_cost"] = 5
    n.generators.loc[n.generators["carrier"] == "oil", "marginal_cost"] = 118.79  #
    n.storage_units.marginal_cost = 74.39
    # display(n.generators.groupby("carrier").sum())

    return n


def clean_generators(network):
    """
    Remove all generators from network
    """
    network.mremove("Generator", [name for name in network.generators.index.tolist()])


def build_redispatch_network(network, network_dispatch):
    '''
    Uses predefined component building functions for building the redispatch network.
    Passes the dispatch per generator as fixed generation for each generator.
    Adds a ramp down generator (neg. redispatch) and ramp up generator (pos. redispatch) for each conventional generator.
    Adds a ramp down generator (curtailment) for each EE generator.
    -----
    Parameters:
    network: network representing one day of snapshots (not optimized)
    network_dispatch: network from dispatch optimization (one node market model)
    '''
    # Copy initial network and remove generators for redispatch step
    network_redispatch = network.copy()
    clean_generators(network_redispatch)

    # Add new generators for redispatch calculation
    l_generators = network.generators.index.tolist()
    l_conv_carriers = ["coal", "lignite", "gas", "oil", "nuclear", "OCGT", "CCGT"]
    l_fluct_renew = ["onwind","offwind-dc", "offwind-ac","solar"]

    for generator in l_generators:
        # For each conventional generator in network: Add 3 generators in network_redispatch
        if network.generators.loc[generator]["carrier"] in l_conv_carriers:
            # Base generator from network, power range is fixed by dispatch simulation results (therefore runs at 0 cost)
            network_redispatch.add("Generator",
                                   name="{}".format(generator),
                                   bus=network.generators.loc[generator]["bus"],
                                   p_nom=network.generators.loc[generator]["p_nom"],
                                   efficiency=network.generators.loc[generator]["efficiency"],
                                   marginal_cost=0,
                                   capital_cost=0,
                                   carrier=network.generators.loc[generator]["carrier"],
                                   p_max_pu=(network_dispatch.generators_t.p[generator] / network_dispatch.generators.loc[generator]["p_nom"]).tolist(),
                                   p_min_pu=(network_dispatch.generators_t.p[generator] / network_dispatch.generators.loc[generator]["p_nom"]).tolist(),
                                   )
            # Upwards generator (positive redispatch) ranges within the difference of the base power p_max_pu (e.g. 0.92) and p_max_pu = 1
            network_redispatch.add("Generator",
                                   name="{}_pos".format(generator),
                                   bus=network.generators.loc[generator]["bus"],
                                   p_nom=network.generators.loc[generator]["p_nom"],
                                   efficiency=network.generators.loc[generator]["efficiency"],
                                   marginal_cost=network.generators.loc[generator]["marginal_cost"],
                                   capital_cost=0,
                                   carrier=network.generators.loc[generator]["carrier"],
                                   p_max_pu=(1 - network_dispatch.generators_t.p[generator] / network_dispatch.generators.loc[generator]["p_nom"]).tolist(),
                                   p_min_pu=0,
                                   )
            # Downwards generator (negative redispatch): Using this generator reduces the energy feed in at specific bus,
            # but increases costs for curtialment. Cost for curtailment = lost profit for not feeding in - savings through ramp down
            network_redispatch.add("Generator",
                                   name="{}_neg".format(generator),
                                   bus=network.generators.loc[generator]["bus"],
                                   p_nom=network.generators.loc[generator]["p_nom"],
                                   efficiency=network.generators.loc[generator]["efficiency"],
                                   marginal_cost=(network.generators.loc[generator]["marginal_cost"] - network_dispatch.buses_t.marginal_price[network_dispatch.generators.loc[generator]["bus"]]).tolist(),
                                   capital_cost=0,
                                   p_max_pu=0,
                                   carrier=network.generators.loc[generator]["carrier"],
                                   p_min_pu=(- network_dispatch.generators_t.p[generator] / network_dispatch.generators.loc[generator]["p_nom"]).tolist(),
                                   )
        elif network.generators.loc[generator]["carrier"] in l_fluct_renew:
                        # For renewable sources: Only add base and negative generator, representing the curtailment of EE
            network_redispatch.add("Generator",
                                   name="{}".format(generator),
                                   bus=network.generators.loc[generator]["bus"],
                                   p_nom=network.generators.loc[generator]["p_nom"],
                                   efficiency=network.generators.loc[generator]["efficiency"],
                                   marginal_cost=0,
                                   capital_cost=0,
                                   carrier=network.generators.loc[generator]["carrier"],
                                   p_max_pu=(network_dispatch.generators_t.p[generator] / network_dispatch.generators.loc[generator]["p_nom"]).tolist(),
                                   p_min_pu=(network_dispatch.generators_t.p[generator] / network_dispatch.generators.loc[generator]["p_nom"]).tolist(),
                                   )
            # Downwards generator (negative redispatch): Using this generator reduces the energy feed in at specific bus,
            # but increases costs for curtialment. Cost for curtailment = lost profit for not feeding in - savings through ramp down
            network_redispatch.add("Generator",
                                   name="{}_neg".format(generator),
                                   bus=network.generators.loc[generator]["bus"],
                                   p_nom=network.generators.loc[generator]["p_nom"],
                                   efficiency=network.generators.loc[generator]["efficiency"],
                                   marginal_cost=(network.generators.loc[generator]["marginal_cost"] - 50 - \
                                                  network_dispatch.buses_t.marginal_price[network_dispatch.generators.loc[generator]["bus"]]).tolist(),
                                   capital_cost=0,
                                   p_max_pu=0,
                                   carrier=network.generators.loc[generator]["carrier"],
                                   p_min_pu=(- network_dispatch.generators_t.p[generator] /
                                             network_dispatch.generators.loc[generator]["p_nom"]).tolist(),
                                   )
        else:
            # For renewable sources: Only add base and negative generator, representing the curtailment of EE
            network_redispatch.add("Generator",
                                   name="{}".format(generator),
                                   bus=network.generators.loc[generator]["bus"],
                                   p_nom=network.generators.loc[generator]["p_nom"],
                                   efficiency=network.generators.loc[generator]["efficiency"],
                                   marginal_cost=0,
                                   capital_cost=0,
                                   carrier=network.generators.loc[generator]["carrier"],
                                   p_max_pu=(network_dispatch.generators_t.p[generator] / network_dispatch.generators.loc[generator]["p_nom"]).tolist(),
                                   p_min_pu=(network_dispatch.generators_t.p[generator] / network_dispatch.generators.loc[generator]["p_nom"]).tolist(),
                                   )
            # Downwards generator (negative redispatch): Using this generator reduces the energy feed in at specific bus,
            # but increases costs for curtialment. Cost for curtailment = lost profit for not feeding in - savings through ramp down
            network_redispatch.add("Generator",
                                   name="{}_neg".format(generator),
                                   bus=network.generators.loc[generator]["bus"],
                                   p_nom=network.generators.loc[generator]["p_nom"],
                                   efficiency=network.generators.loc[generator]["efficiency"],
                                   marginal_cost=(network.generators.loc[generator]["marginal_cost"] -
                                                  network_dispatch.buses_t.marginal_price[network_dispatch.generators.loc[generator]["bus"]]).tolist(),
                                   capital_cost=0,
                                   p_max_pu=0,
                                   carrier=network.generators.loc[generator]["carrier"],
                                   p_min_pu=(- network_dispatch.generators_t.p[generator] /
                                             network_dispatch.generators.loc[generator]["p_nom"]).tolist(),
                                   )
    # add load shedding
    bus_names = network_redispatch.buses.index.tolist()
    for i in range(len(bus_names)):
        bus_name = bus_names[i]
        # add load shedding generator at every bus
        network_redispatch.add("Generator",
                                name="load_{}".format(i + 1),
                                carrier = "load",
                                bus = bus_name,
                                p_nom = 50000, # in pypsa-eur: 1*10^9 KW, marginal cost auch per kW angegeben
                                efficiency = 1,
                                marginal_cost = 5000, # non zero marginal cost to ensure unique optimization result
                                capital_cost = 0,
                                p_nom_extendable = False,
                                p_nom_min = 0,
                                p_nom_max = math.inf)

    network_redispatch.name = str(network_redispatch.name) + " redispatch"
    return network_redispatch


def build_market_model(network):
    '''
    Clusters all generators, loads, stores and storage units to a single (medoid) bus to emulate market simulation.
    For faster computation: Assign to first bus of dataset
    TODO: For multiple countries, the dataset to be clusterd first needs to be separated by country (groupby("country")) and then clustered.
    '''
    mode = "default"
    network_dispatch = network.copy()

    # cluster all buses to the center one and assign all grid elements to that bus
    df_busmap = network_dispatch.buses

    if mode == "cluster":
        # Cluster buses to 1 cluster & compute center
        np_busmap = df_busmap[["x", "y"]].to_numpy()
        k_medoids = KMedoids(n_clusters=1, metric='euclidean', init='k-medoids++', max_iter=300, random_state=None)
        score = k_medoids.fit(np_busmap)
        x_medoid = k_medoids.cluster_centers_[0][0]
        y_medoid = k_medoids.cluster_centers_[0][1]
        # find single medoid bus in dataframe
        market_bus = df_busmap[(df_busmap["x"] == x_medoid) & (df_busmap["y"] == y_medoid)].squeeze()
        bus_name = str(market_bus.name)  # save bus name, remove brackets & quotemarks
    else:
        # bus_name = str(df_busmap[0].index.values)[2:-2]
        bus_name = df_busmap.iloc[0].name

    # assign all generators, loads, storage_units, stores to the medoid bus (market_bus)
    network_dispatch.loads["bus"] = bus_name
    network_dispatch.generators["bus"] = bus_name
    network_dispatch.stores["bus"] = bus_name
    network_dispatch.storage_units["bus"] = bus_name
    return network_dispatch


def solve_redispatch_network(network, network_dispatch):
    '''
    Calls the building function and solves the network under redispatch constraints.
    -----
    Parameters:
    network: network from dispatch optimization
    '''
    network_redispatch = build_redispatch_network(network, network_dispatch)
    network_redispatch.lopf(solver_name="gurobi", pyomo=True, formulation="kirchhoff")
    return network_redispatch


def concat_network(list_networks, ignore_standard_types=False):
    '''
    Function that merges technically identical, but temporally decoupled networks by concatenating
    their time-dependent components (input & output)
    -----
    Param:
    l_networks: list of daily solved networks (can be either l_networks_dispatch or l_networks_redispatch)
    scenario: Determines whether input networks will have stores or not
    '''
    from pypsa.io import (import_components_from_dataframe, import_series_from_dataframe)
    from six import iterkeys

    # create new network out of first network of the list of identical networks
    n_input = list_networks[0].copy()

    # Copy time indipendent components
    # -------------------
    override_components, override_component_attrs = n_input._retrieve_overridden_components()
    nw = n_input.__class__(ignore_standard_types=ignore_standard_types,
                           override_components=override_components,
                           override_component_attrs=override_component_attrs)

    for component in n_input.iterate_components(
            ["Bus", "Carrier"] + sorted(n_input.all_components - {"Bus", "Carrier"})):
        df = component.df
        # drop the standard types to avoid them being read in twice
        if not ignore_standard_types and component.name in n_input.standard_type_components:
            df = component.df.drop(nw.components[component.name]["standard_types"].index)
        import_components_from_dataframe(nw, df, component.name)

    # Time dependent components
    # --------------------
    # set snapshots
    snapshots = n_input.snapshots
    for network in list_networks[1:]:
        snapshots = snapshots.union(network.snapshots)
    nw.set_snapshots(snapshots)

    # concat time dependent components from all networks in input list
    for component in nw.iterate_components(["Bus", "Carrier"] + sorted(n_input.all_components - {"Bus", "Carrier"})):
        component_t = component.list_name + "_t"

        for attr, timeseries in component.pnl.items():
            l_component = []
            for network in list_networks:
                # each time dependent dataframe
                l_component.append(getattr(getattr(network, component_t), attr))
            # concat the components list to dataframe
            df_component = pd.concat(l_component, axis=0)
            # import time series from dataframe for output network
            import_series_from_dataframe(nw, df_component, component.name, attr)

    # catch all remaining attributes of network
    for attr in ["name", "srid"]:
        setattr(nw, attr, getattr(n_input, attr))

        # Concat objective value for partially solved networks
    obj = 0
    for network in list_networks:
        if hasattr(network, 'objective'):
            obj = obj + network.objective
    nw.objective = obj
    return nw


def redispatch_workflow(n, n_optim, scenario, ratio_wind, ratio_pv, sec_margin):
    '''
    Function for executing the whole redispatch workflow.

    -----
    Param:
    network: Un-optimized network from pypsa-eur.
    network_optim: Already optimized network from the strategic network optimization in pypsa-eur.
    scenario: "bat" or "no bat" decides whether the redispatch optimization is run with or without batteries.
    Return:
    l_networks_list: List of solved daily dispatch and redispatch networks.
    '''

    # Make a copy of the unsolved full year network
    network_year = n.copy()
    network = n.copy()
    network_optim = n_optim.copy()
    # create lists to save results
    print(sec_margin)
    l_networks_24 = []
    l_networks_dispatch = []
    l_networks_redispatch = []
    start_hour_0 = 0

    # Generate operative pypsa-eur network without investment problem
    network = set_parameters_from_optimized(n=network, n_optim=network_optim, ratio_wind=ratio_wind, ratio_pv=ratio_pv, sec_margin=sec_margin)

    # Only operative optimization: Capital costs set to zero
    network.generators.loc[:, "capital_cost"] = 0

    # Run dispatch_redispatch workflow for every day (24h batch)
    for start_hour in range(start_hour_0, len(network.snapshots), 24):

        # print(start_hour)
        n_24 = network.copy(snapshots=network.snapshots[start_hour:start_hour + 24])

        # start SOC of storage_units equals state of charge at last snapshot of previous day
        n_24.storage_units["cyclic_state_of_charge"] = True
        print("\n\n\n")
        print(n_24.generators.groupby("carrier").p_nom.sum())
        print("\n\n\n")

        # Build market model, solve dispatch network and plot network insights
        # ---------------
        n_dispatch = build_market_model(n_24)
        n_dispatch.lopf(solver_name="gurobi", pyomo=True, formulation="kirchhoff")

        # Call redispatch optimization and plot network insights
        # ---------------
        if scenario == "no bat":
            n_redispatch = solve_redispatch_network(n_24, n_dispatch)

        # Append networks to corresponding yearly lists
        # ---------------
        l_networks_24.append(n_24)
        l_networks_dispatch.append(n_dispatch)
        l_networks_redispatch.append(n_redispatch)

    # For each results list: Create a network out of daily networks for dispatch & redispatch
    # ---------------
    network_dispatch = concat_network(l_networks_dispatch)
    network_redispatch = concat_network(l_networks_redispatch)

    return network_dispatch, network_redispatch


def solve_all_redispatch_workflows(c_rate=0.25, flex_share=0.1):
    """
    Function to run the redispatch workflow for all networks in the networks_redispatch folder.
    This function is used to compare multiple historic redispatch networks (without bat) whether the redispatch changes
    drastically. It is run without batteries.
    """

    folder = r'/cluster/home/wlaumen/Euler/pypsa-eur/networks_redispatch'
    for filepath in glob.iglob(folder + '/*.nc'):
        filename = filepath.split('/')[-1].split(".")[0]
        print(filename + "\n\n\n\n\n\n")
        path_n = filepath
        path_n_optim = folder + "/solved/" + filename + ".nc"
        # Define network and network_optim

        n = pypsa.Network(path_n)
        n_optim = pypsa.Network(path_n_optim)

        sec_margin = 1
        #Run redispatch w/o batteries & export files
        n_d, n_rd = redispatch_workflow(n, n_optim, scenario="no bat", ratio_wind = 2.2, ratio_pv = 1.38, sec_margin=sec_margin)
        # export solved dispatch & redispatch workflow as well as objective value list
        export_path = folder + r"/results"
        n_d.export_to_netcdf(path=export_path + r"/dispatch/" + filename + "_1_coal_50wind.nc", export_standard_types=False, least_significant_digit=None)
        n_rd.export_to_netcdf(path=export_path + r"/redispatch/" + filename + "_1_coal_50wind.nc", export_standard_types=False, least_significant_digit=None)
        gc.collect()

def main():
    solve_all_redispatch_workflows(c_rate=0.25, flex_share=0.1)
    print("Test")
if __name__ == "__main__":
    main()
