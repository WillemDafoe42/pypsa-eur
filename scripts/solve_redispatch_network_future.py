# Redispatch workflow future networks

import pypsa
import matplotlib.pyplot as plt
import math
import pandas as pd
import gc
import numpy as np
import pickle as pkl
import glob


def clean_batteries(network):
    """
    Clean up all previously saved battery components (FROM PYPSA-EUR) related to redispatch (does NOT delete the inital batteries)
    """
    #     network.mremove("StorageUnit", [name for name in network.storage_units.index.tolist()
    #                                     if network.storage_units.loc[name]["carrier"] == "battery"])
    network.mremove("Link", [name for name in network.links.index.tolist()
                             if "BESS" in name])
    network.mremove("Store", [name for name in network.stores.index.tolist()
                              if "BESS" in name])
    network.mremove("Bus", [name for name in network.buses.index.tolist()
                            if "BESS" in name])


def add_BESS_load(network, network_dispatch, network_year, start_hour, start_hour_0, c_rate, flex_potential,
                  s_SOC_batteries, lcos, n_dispatch_prev):
    '''
    Adds battery storages at every node, depending on the flexibility potential of the loads allocated to this node.
    Methodology according to Network development plan and Frontier study:
        - Flexibility potential at load-nodes: Flexibility Potential through stationary battery storage in distribution grids
        - Flexibility potential at power plants: Flexibility through battery installations at powerplants (by operators)

    Following the assumption of x percentage of frontier flexibility potential being installed at node according to loads.
    Flexibility potential can e.g. be aggregated powerwall capacity at the node.

    Update: Funktion für die batteriekosten jetzt (für positiven redispatch): LCOS + mean strompreis der letzten 24 stunden
            (24 stunden aus dispatch networks)
            (Für negativen Redispatch):

    Sources: Alex paper, frontier study
    '''
    print("add_BESS_load")
    print(str(lcos) + "\n\n")

    # Flexibility potential

    # Clean up all previously saved battery components
    clean_batteries(network)

    # get bus names
    bus_names = network.buses.index.tolist()
    # get mean load at every bus
    df_loads = network_year.loads.copy()
    df_loads["p_mean"] = network_year.loads_t.p_set.mean(axis=0)
    p_mean_sum = df_loads["p_mean"].sum()

    # Get mean market price of last 24h to estimate future DISCHARGING profit for battery operator that provides negative redispatch
    # (has to charge energy). For positive redispatch: it is assumed that the battery operator has to recharge at market price,
    # so he will try to charge at smallest possible price (not always possible to charge at total minimum)
    # Get mean market price of last 24h, if lcos is 0, run at no cost (no charging cost as well) for getting theoretical potential
    if lcos == 0:
        mean_price_24h = 1
        charge_price_24h = 0
    else:
        if type(n_dispatch_prev) != int:
            min_today = network_dispatch.buses_t.marginal_price.mean(axis=1).nsmallest(n=4)
            min_prev = n_dispatch_prev.buses_t.marginal_price.mean(axis=1).nsmallest(n=4)
            min_price = pd.concat((min_today, min_prev), axis=0).nsmallest(n=4).mean()
        else:
            min_price = network_dispatch.buses_t.marginal_price.mean(axis=1).nsmallest(n=4).mean()
        mean_price_24h = min(network_dispatch.buses_t.marginal_price.mean(axis=1).mean(),
                             network_dispatch.generators[
                                 network_dispatch.generators["carrier"] == "lignite"].marginal_cost.mean()) - 1
        charge_price_24h = min(min_price, network_dispatch.generators[
            network_dispatch.generators["carrier"] == "lignite"].marginal_cost.mean())

    # According to x% rule, add energy stores with energy = x% of average load at bus
    for i in range(len(bus_names)):
        bus_name = bus_names[i]
        battery_bus = "{}_{}".format(bus_name, "BESS")

        # determine flexibility at bus: Total flexibility at bus equals share of mean load of aggregate mean load
        df_load_bus = df_loads[df_loads["bus"] == bus_name]
        p_mean_bus = df_load_bus["p_mean"].iloc[0] if not df_load_bus.empty else 0
        flex_share = p_mean_bus / p_mean_sum
        p_flex = flex_potential * flex_share

        # add additional bus solely for representation of battery at location of previous bus
        network.add("Bus", name=battery_bus, x=network.buses.loc[bus_name, "x"], y=network.buses.loc[bus_name, "y"],
                    carrier="battery")
        # add store
        network.add("Store", name="BESS_{}".format(i), bus=battery_bus,
                    e_nom=p_flex / c_rate, e_nom_extendable=False,
                    e_min_pu=0, e_initial=p_flex / c_rate * 0.5, e_cyclic=False,
                    p_set=p_flex, q_set=0.05, marginal_cost=0,
                    capital_cost=0, standing_loss=0)
        # discharge link with LCOS cost
        network.add("Link", name="BESS_{}_discharger".format(i),
                    bus0=battery_bus, bus1=bus_name, capital_cost=0,
                    p_nom=p_flex, p_nom_extendable=False, p_max_pu=1, p_min_pu=0,
                    marginal_cost=lcos + charge_price_24h, efficiency=0.96)
        # charge link: Ensure that southern batteries might be charged before northern ones, by making their marginal cost slightly cheaper
        if network.buses.loc[bus_name, "y"] < 51:
            network.add("Link", name="BESS_{}_charger".format(i),
                        bus0=bus_name, bus1=battery_bus, capital_cost=0,
                        p_nom=p_flex, p_nom_extendable=False, p_max_pu=1, p_min_pu=0,
                        marginal_cost=mean_price_24h * (-1) - 1, efficiency=0.96)
        else:
            network.add("Link", name="BESS_{}_charger".format(i),
                        bus0=bus_name, bus1=battery_bus, capital_cost=0,
                        p_nom=p_flex, p_nom_extendable=False, p_max_pu=1, p_min_pu=0,
                        marginal_cost=mean_price_24h * (-1), efficiency=0.96)

    # If not first day: Set initial SOC of batteries to end soc of previous day.
    if start_hour != start_hour_0:
        network.stores["e_initial"] = s_SOC_batteries
    return network

def add_BESS_supply(network, network_dispatch, network_year, start_hour, start_hour_0, c_rate, plant_potential,
                    s_SOC_batteries, lcos, n_dispatch_prev):
    '''
    Adds battery storages at every node, depending on the flexibility potential of the Powerplants allocated to this node.
    Methodology according to Network development plan and Frontier study:
        - Flexibility potential at power plants: Flexibility through battery installations at powerplants (by operators)

    -----
    Param:
    network: network from dispatch optimization
    network_year: full time resolution network (typically 1 year)
    s_SOC_batteries: Series of battery SOCs from LAST hour of previous day. Dafault = empty pandas series (for first day)
    '''
    print("\n\nsupply\n\n")
    # Clean up all previously saved battery components
    clean_batteries(network)

    # get bus names
    bus_names = network.buses.index.tolist()

    # Get mean market price of last 24h, if lcos is 0, run at no cost (no charging cost as well) for getting theoretical potential
    if lcos == 0:
        mean_price_24h = 1
        charge_price_24h = 0
    else:
        if type(n_dispatch_prev) != int:
            min_today = network_dispatch.buses_t.marginal_price.mean(axis=1).nsmallest(n=4)
            min_prev = n_dispatch_prev.buses_t.marginal_price.mean(axis=1).nsmallest(n=4)
            min_price = pd.concat((min_today, min_prev), axis=0).nsmallest(n=4).mean()
        else:
            min_price = network_dispatch.buses_t.marginal_price.mean(axis=1).nsmallest(n=4).mean()
        mean_price_24h = min(network_dispatch.buses_t.marginal_price.mean(axis=1).mean(),
                             network_dispatch.generators[
                                 network_dispatch.generators["carrier"] == "lignite"].marginal_cost.mean()) - 1
        charge_price_24h = min(min_price, network_dispatch.generators[
            network_dispatch.generators["carrier"] == "lignite"].marginal_cost.mean())

    # Add supply side storage at power plants according to NDP goal
    # ----------------
    min_gen_size = 100

    # Total large scale plant capcity (generators > 100)
    df_gen = network_year.generators[["p_nom_max", "p_nom", "bus"]].copy()
    df_gen["flex_plant"] = df_gen["p_nom"].apply(lambda x: x if x > min_gen_size else 0)
    p_gen_sum = df_gen["flex_plant"].sum()

    for i in range(len(bus_names)):
        bus_name = bus_names[i]
        gen_battery_bus = "{}_{}".format(bus_name, "generator_BESS")

        # determine powerplant sided storage: share of generation capacity of generation capacity added as storage
        # ONLY if generation cap exceeds threshhold -> NDP: Large scale generation shall have storage for RD
        # Adds flexibility for each generator if p_nom > 100 -> flexibility is then aggregated to one storage
        df_gen_bus = df_gen[df_gen["bus"] == bus_name]

        p_flex_plant = df_gen_bus["flex_plant"].sum()
        p_flex_plant = plant_potential * p_flex_plant / p_gen_sum

        if p_flex_plant > 0:
            # add additional bus for battery at location of previous bus
            network.add("Bus", name=gen_battery_bus,
                        x=network.buses.loc[bus_name, "x"], y=network.buses.loc[bus_name, "y"], carrier="battery")
            # add store
            network.add("Store", name="generator_BESS_{}".format(i), bus=gen_battery_bus,
                        e_nom=p_flex_plant / c_rate, e_nom_extendable=False,
                        e_min_pu=0, e_max_pu=1, e_initial=p_flex_plant / c_rate * 0.5, e_cyclic=False,
                        p_set=p_flex_plant,
                        q_set=0.05, marginal_cost=0,
                        capital_cost=0, standing_loss=0)
            # discharge link
            network.add("Link", name="generator_BESS_{}_discharger".format(i),
                        bus0=gen_battery_bus, bus1=bus_name, capital_cost=0,
                        p_nom=p_flex_plant, p_nom_extendable=False, p_max_pu=1, p_min_pu=0,
                        marginal_cost=lcos + charge_price_24h, efficiency=0.96)
            # charge link
            if network.buses.loc[bus_name, "y"] < 51:
                network.add("Link", name="generator_BESS_{}_charger".format(i),
                            bus0=bus_name, bus1=gen_battery_bus, capital_cost=0,
                            p_nom=p_flex_plant, p_nom_extendable=False, p_max_pu=1, p_min_pu=0,
                            marginal_cost=mean_price_24h * (-1) - 1, efficiency=0.96)
            else:
                network.add("Link", name="generator_BESS_{}_charger".format(i),
                            bus0=bus_name, bus1=gen_battery_bus, capital_cost=0,
                            p_nom=p_flex_plant, p_nom_extendable=False, p_max_pu=1, p_min_pu=0,
                            marginal_cost=mean_price_24h * (-1), efficiency=0.96)

                # If not first day: Set initial SOC of batteries to end soc of previous day.
    if start_hour != start_hour_0:
        network.stores["e_initial"] = s_SOC_batteries
    return network

def add_BESS_all(network, network_dispatch, network_year, start_hour, start_hour_0, c_rate,
                 flex_potential, plant_potential, s_SOC_batteries, lcos, n_dispatch_prev):
    '''
    Adds battery storages at every node, depending on the flexibility potential of the loads allocated to this node.
    Methodology according to Network development plan and Frontier study:
        - Flexibility potential at load-nodes: Flexibility Potential through stationary battery storage in distribution grids
        - Flexibility potential at power plants: Flexibility through battery installations at powerplants (by operators)

    Following the assumption of x percentage of frontier flexibility potential being installed at node according to loads.
    Flexibility potential can e.g. be aggregated powerwall capacity at the node.
    -----
    Param:
    network: network from dispatch optimization
    network_year: full time resolution network (typically 1 year)
    flex_potential: Total flexibility potential from distribution grid (default: frontier-study potential)
    s_SOC_batteries: Series of battery SOCs from LAST hour of previous day. Dafault = empty pandas series (for first day)
    '''
    print("\n\nall\n\n")

    # Clean up all previously saved battery components
    clean_batteries(network)

    # get bus names
    bus_names = network.buses.index.tolist()
    # get mean load at every bus
    df_loads = network_year.loads.copy()
    df_loads["p_mean"] = network_year.loads_t.p_set.mean(axis=0)
    p_mean_sum = df_loads["p_mean"].sum()

    # Get mean market price of last 24h
    # Get mean market price of last 24h, if lcos is 0, run at no cost (no charging cost as well) for getting theoretical potential
    if lcos == 0:
        mean_price_24h = 1
        charge_price_24h = 0
    else:
        if type(n_dispatch_prev) != int:
            min_today = network_dispatch.buses_t.marginal_price.mean(axis=1).nsmallest(n=4)
            min_prev = n_dispatch_prev.buses_t.marginal_price.mean(axis=1).nsmallest(n=4)
            min_price = pd.concat((min_today, min_prev), axis=0).nsmallest(n=4).mean()
        else:
            min_price = network_dispatch.buses_t.marginal_price.mean(axis=1).nsmallest(n=4).mean()
        mean_price_24h = min(network_dispatch.buses_t.marginal_price.mean(axis=1).mean(),
                             network_dispatch.generators[network_dispatch.generators["carrier"] == "lignite"].marginal_cost.mean()) - 1
        charge_price_24h = min(min_price, network_dispatch.generators[
            network_dispatch.generators["carrier"] == "lignite"].marginal_cost.mean())

    # According to x% rule, add energy stores with energy = x% of average load at bus
    for i in range(len(bus_names)):
        bus_name = bus_names[i]
        battery_bus = "{}_{}".format(bus_name, "BESS")

        # determine flexibility at bus: Total flexibility at bus equals share of mean load of aggregate mean load
        df_load_bus = df_loads[df_loads["bus"] == bus_name]
        p_mean_bus = df_load_bus["p_mean"].iloc[0] if not df_load_bus.empty else 0
        flex_share = p_mean_bus / p_mean_sum
        p_flex = flex_potential * flex_share

        # add additional bus solely for representation of battery at location of previous bus
        network.add("Bus", name=battery_bus, x=network.buses.loc[bus_name, "x"], y=network.buses.loc[bus_name, "y"],
                    carrier="battery")
        # add store
        network.add("Store", name="BESS_{}".format(i), bus=battery_bus,
                    e_nom=p_flex / c_rate, e_nom_extendable=False,
                    e_min_pu=0,
                    e_initial=p_flex / c_rate * 0.5, e_cyclic=False,
                    p_set=p_flex, q_set=0.05, marginal_cost=0,
                    capital_cost=0, standing_loss=0)
        # discharge link with LCOS cost
        network.add("Link", name="BESS_{}_discharger".format(i),
                    bus0=battery_bus, bus1=bus_name, capital_cost=0,
                    p_nom=p_flex, p_nom_extendable=False, p_max_pu=1, p_min_pu=0,
                    marginal_cost=lcos + charge_price_24h, efficiency=0.96)
        # charge link
        if network.buses.loc[bus_name, "y"] < 51:
            network.add("Link", name="BESS_{}_charger".format(i),
                        bus0=bus_name, bus1=battery_bus, capital_cost=0,
                        p_nom=p_flex, p_nom_extendable=False, p_max_pu=1, p_min_pu=0,
                        marginal_cost=mean_price_24h * (-1) - 1, efficiency=0.96)
        else:
            network.add("Link", name="BESS_{}_charger".format(i),
                        bus0=bus_name, bus1=battery_bus, capital_cost=0,
                        p_nom=p_flex, p_nom_extendable=False, p_max_pu=1, p_min_pu=0,
                        marginal_cost=mean_price_24h * (-1), efficiency=0.96)

    # Add supply side storage at power plants according to NDP goal
    # ----------------
    min_gen_size = 100

    # Total large scale plant capcity (generators > 100)
    df_gen = network_year.generators[["p_nom_max", "p_nom", "bus"]].copy()
    df_gen["flex_plant"] = df_gen["p_nom"].apply(lambda x: x if x > min_gen_size else 0)
    p_gen_sum = df_gen["flex_plant"].sum()

    for i in range(len(bus_names)):
        bus_name = bus_names[i]
        gen_battery_bus = "{}_{}".format(bus_name, "generator_BESS")

        # determine powerplant sided storage: share of generation capacity of generation capacity added as storage
        # ONLY if generation cap exceeds threshhold -> NDP: Large scale generation shall have storage for RD
        # Adds flexibility for each generator if p_nom > 100 -> flexibility is then aggregated to one storage
        df_gen_bus = df_gen[df_gen["bus"] == bus_name]

        p_flex_plant = df_gen_bus["flex_plant"].sum()
        p_flex_plant = plant_potential * p_flex_plant / p_gen_sum

        if p_flex_plant > 0:
            # add additional bus for battery at location of previous bus
            network.add("Bus", name=gen_battery_bus,
                        x=network.buses.loc[bus_name, "x"], y=network.buses.loc[bus_name, "y"], carrier="battery")
            # add store
            network.add("Store", name="generator_BESS_{}".format(i), bus=gen_battery_bus,
                        e_nom=p_flex_plant / c_rate, e_nom_extendable=False,
                        e_min_pu=0, e_max_pu=1, e_initial=p_flex_plant / c_rate * 0.5,
                        e_cyclic=False, p_set=p_flex_plant,
                        q_set=0.05, marginal_cost=0,
                        capital_cost=0, standing_loss=0)
            # discharge link
            network.add("Link", name="generator_BESS_{}_discharger".format(i),
                        bus0=gen_battery_bus, bus1=bus_name, capital_cost=0,
                        p_nom=p_flex_plant, p_nom_extendable=False, p_max_pu=1, p_min_pu=0,
                        marginal_cost=lcos + charge_price_24h, efficiency=0.96)
            # charge link
            if network.buses.loc[bus_name, "y"] < 51:
                network.add("Link", name="generator_BESS_{}_charger".format(i),
                            bus0=bus_name, bus1=gen_battery_bus, capital_cost=0,
                            p_nom=p_flex_plant, p_nom_extendable=False, p_max_pu=1, p_min_pu=0,
                            marginal_cost=mean_price_24h * (-1) - 1, efficiency=0.96)
            else:
                network.add("Link", name="generator_BESS_{}_charger".format(i),
                            bus0=bus_name, bus1=gen_battery_bus, capital_cost=0,
                            p_nom=p_flex_plant, p_nom_extendable=False, p_max_pu=1, p_min_pu=0,
                            marginal_cost=mean_price_24h * (-1), efficiency=0.96)

                # If not first day: Set initial SOC of batteries to end soc of previous day.
    if start_hour != start_hour_0:
        network.stores["e_initial"] = s_SOC_batteries
    return network


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
    l_fluct_renew = ["onwind", "offwind-dc", "offwind-ac", "solar"]

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
                                   p_max_pu=(network_dispatch.generators_t.p[generator] /
                                             network_dispatch.generators.loc[generator]["p_nom"]).tolist(),
                                   p_min_pu=(network_dispatch.generators_t.p[generator] /
                                             network_dispatch.generators.loc[generator]["p_nom"]).tolist(),
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
                                   p_max_pu=(1 - network_dispatch.generators_t.p[generator] /
                                             network_dispatch.generators.loc[generator]["p_nom"]).tolist(),
                                   p_min_pu=0,
                                   )
            # Downwards generator (negative redispatch): Using this generator reduces the energy feed in at specific bus,
            # but increases costs for curtialment. Cost for curtailment = lost profit for not feeding in - savings through ramp down
            network_redispatch.add("Generator",
                                   name="{}_neg".format(generator),
                                   bus=network.generators.loc[generator]["bus"],
                                   p_nom=network.generators.loc[generator]["p_nom"],
                                   efficiency=network.generators.loc[generator]["efficiency"],
                                   marginal_cost=(network.generators.loc[generator]["marginal_cost"] -
                                                  network_dispatch.buses_t.marginal_price[
                                                      network_dispatch.generators.loc[generator]["bus"]]).tolist(),
                                   capital_cost=0,
                                   p_max_pu=0,
                                   carrier=network.generators.loc[generator]["carrier"],
                                   p_min_pu=(- network_dispatch.generators_t.p[generator] /
                                             network_dispatch.generators.loc[generator]["p_nom"]).tolist(),
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
                                   p_max_pu=(network_dispatch.generators_t.p[generator] /
                                             network_dispatch.generators.loc[generator]["p_nom"]).tolist(),
                                   p_min_pu=(network_dispatch.generators_t.p[generator] /
                                             network_dispatch.generators.loc[generator]["p_nom"]).tolist(),
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
                                   p_max_pu=(abs(network_dispatch.generators_t.p_max_pu[generator].round(2) - (network_dispatch.generators_t.p[generator] /
                                               network_dispatch.generators.loc[generator]["p_nom"]).round(2))).tolist(),
                                   p_min_pu=0,
                                   )
            # Downwards generator (negative redispatch): Using this generator reduces the energy feed in at specific bus,
            # but increases costs for curtialment. Cost for curtailment = lost profit for not feeding in - savings through ramp down
            network_redispatch.add("Generator",
                                   name="{}_neg".format(generator),
                                   bus=network.generators.loc[generator]["bus"],
                                   p_nom=network.generators.loc[generator]["p_nom"],
                                   efficiency=network.generators.loc[generator]["efficiency"],
                                   marginal_cost=(network.generators.loc[generator]["marginal_cost"] -
                                                  network_dispatch.buses_t.marginal_price[
                                                      network_dispatch.generators.loc[generator]["bus"]]).tolist(),
                                   capital_cost=0,
                                   p_max_pu=0,
                                   carrier=network.generators.loc[generator]["carrier"],
                                   p_min_pu=(- network_dispatch.generators_t.p[generator] /
                                             network_dispatch.generators.loc[generator]["p_nom"]).tolist()
                                   )
            # display(abs(network_dispatch.generators_t.p_max_pu[generator].round(2)))
            # display(abs((network_dispatch.generators_t.p[generator] / network_dispatch.generators.loc[generator]["p_nom"]).round(2)))
            # display(abs((network_dispatch.generators_t.p_max_pu[generator]-network_dispatch.generators_t.p[generator] / network_dispatch.generators.loc[generator]["p_nom"]).round(2)))


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
                                   p_max_pu=(network_dispatch.generators_t.p[generator] /
                                             network_dispatch.generators.loc[generator]["p_nom"]).tolist(),
                                   p_min_pu=(network_dispatch.generators_t.p[generator] /
                                             network_dispatch.generators.loc[generator]["p_nom"]).tolist(),
                                   )
            # Downwards generator (negative redispatch): Using this generator reduces the energy feed in at specific bus,
            # but increases costs for curtialment. Cost for curtailment = lost profit for not feeding in - savings through ramp down
            network_redispatch.add("Generator",
                                   name="{}_neg".format(generator),
                                   bus=network.generators.loc[generator]["bus"],
                                   p_nom=network.generators.loc[generator]["p_nom"],
                                   efficiency=network.generators.loc[generator]["efficiency"],
                                   marginal_cost=(network.generators.loc[generator]["marginal_cost"] -
                                                  network_dispatch.buses_t.marginal_price[
                                                      network_dispatch.generators.loc[generator]["bus"]]).tolist(),
                                   capital_cost=0,
                                   p_max_pu=0,
                                   carrier=network.generators.loc[generator]["carrier"],
                                   p_min_pu=(- network_dispatch.generators_t.p[generator] /
                                             network_dispatch.generators.loc[generator]["p_nom"]).tolist()
                                   )

    # add load shedding
    bus_names = network_redispatch.buses.index.tolist()
    for i in range(len(bus_names)):
        bus_name = bus_names[i]
        # add load shedding generator at every bus
        network_redispatch.add("Generator",
                               name="load_{}".format(i + 1),
                               carrier="load",
                               bus=bus_name,
                               p_nom=100000,  # in pypsa-eur: 1*10^9 KW, marginal cost auch per kW angegeben
                               efficiency=1,
                               marginal_cost=1000,  # non zero marginal cost to ensure unique optimization result
                               capital_cost=0,
                               p_nom_extendable=False,
                               p_nom_min=0,
                               p_nom_max=math.inf)

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

    # add load shedding generator at market bus
    network_dispatch.add("Generator",
                           name="load",
                           carrier="load",
                           bus=bus_name,
                           p_nom=100000,  # in pypsa-eur: 1*10^9 KW, marginal cost auch per kW angegeben
                           efficiency=1,
                           marginal_cost=network_dispatch.generators[network_dispatch.generators["carrier"]=="oil"].marginal_cost.mean() + 0.1,  # non zero marginal cost to ensure unique optimization result
                           capital_cost=0,
                           p_nom_extendable=False,
                           p_nom_min=0,
                           p_nom_max=math.inf)

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


# Redispatch workflow with batteries
def build_redispatch_network_with_bat(network, network_dispatch, network_year, start_hour, start_hour_0,
                                      c_rate, storage_ops, flex_potential, plant_potential, s_SOC_batteries,
                                      lcos, n_dispatch_prev):
    '''
    Uses predefined component building functions for building the redispatch network and adds batteries.
    -----
    Param:
    network: network from dispatch optimization
    '''
    network_redispatch_bat = build_redispatch_network(network, network_dispatch)
    if storage_ops == "load":
        network_redispatch_bat= add_BESS_load(network_redispatch_bat, network_dispatch, network_year=network_year, start_hour=start_hour,
                      start_hour_0=start_hour_0, c_rate=c_rate,
                      flex_potential=flex_potential, s_SOC_batteries=s_SOC_batteries, lcos=lcos, n_dispatch_prev=n_dispatch_prev)
    elif storage_ops == "supply":
        network_redispatch_bat= add_BESS_supply(network_redispatch_bat, network_dispatch, network_year=network_year, start_hour=start_hour,
                        start_hour_0=start_hour_0, c_rate=c_rate,
                        plant_potential=plant_potential, s_SOC_batteries=s_SOC_batteries, lcos=lcos, n_dispatch_prev=n_dispatch_prev)
    elif storage_ops == "all":
        network_redispatch_bat= add_BESS_all(network_redispatch_bat, network_dispatch, network_year=network_year, start_hour=start_hour,
                     start_hour_0=start_hour_0, c_rate=c_rate,
                     flex_potential=flex_potential, plant_potential=plant_potential, s_SOC_batteries=s_SOC_batteries,
                     lcos=lcos, n_dispatch_prev=n_dispatch_prev)
    return network_redispatch_bat


def solve_redispatch_network_with_bat(network, network_dispatch, network_year, start_hour, start_hour_0,
                                      c_rate, storage_ops, flex_potential, plant_potential, s_SOC_batteries, lcos, n_dispatch_prev):
    '''
    Calls redispatch building function and solves the network.
    -----
    Parameters:
    network: network from dispatch optimization
    '''
    network_redispatch_bat = build_redispatch_network_with_bat(network=network, network_dispatch=network_dispatch,
                                                               network_year=network_year, start_hour=start_hour,
                                                               start_hour_0=start_hour_0, c_rate=c_rate,
                                                               storage_ops=storage_ops, flex_potential=flex_potential,
                                                               plant_potential=plant_potential,
                                                               s_SOC_batteries=s_SOC_batteries, lcos=lcos,
                                                               n_dispatch_prev=n_dispatch_prev)
    network_redispatch_bat.lopf(solver_name="gurobi", pyomo=True, formulation="kirchhoff")

    return network_redispatch_bat


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


def redispatch_workflow_future(n_future, c_rate, flex_potential, plant_potential, storage_ops, scenario,
                               ratio_wind, ratio_pv, lcos):
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

    # Make a copy of the unsolved full year future network
    network = n_future.copy()
    network_year = n_future.copy()
    l_networks_24 = []
    l_networks_dispatch = []
    l_networks_redispatch = []
    #start_hour_0 = 7848
    start_hour_0 = 6570
    # start_hour_0 = 0

    # Only operative optimization: Capital costs set to zero
    network.generators.loc[:, "capital_cost"] = 0

    # Run dispatch_redispatch workflow for every day (24h batch)
    # for start_hour in range(start_hour_0, start_hour_0 + 7*24, 24):

    for start_hour in range(start_hour_0, len(network.snapshots), 24): # len(network.snapshots)

        # print(start_hour)
        n_24 = network.copy(snapshots=network.snapshots[start_hour:start_hour + 24])

        # start SOC of storage_units equals state of charge at last snapshot of previous day
        n_24.storage_units["cyclic_state_of_charge"] = True

        # Build market model, solve dispatch network and plot network insights
        # ---------------
        print("\n\nNow Dispatch\n\n")
        n_dispatch = build_market_model(n_24)
        n_dispatch.lopf(solver_name="gurobi", pyomo=True, formulation="kirchhoff")

        # Call redispatch optimization and plot network insights
        # ---------------
        print("\n\nNow Redispatch\n\n")
        if scenario == "no bat":
            n_redispatch = solve_redispatch_network(n_24, n_dispatch)
        elif scenario == "bat" and start_hour == start_hour_0:
            n_redispatch = solve_redispatch_network_with_bat(n_24, n_dispatch, network_year,
                                                             start_hour=start_hour, start_hour_0=start_hour_0,
                                                             c_rate=c_rate, flex_potential=flex_potential,
                                                             plant_potential=plant_potential, storage_ops=storage_ops,
                                                             s_SOC_batteries=pd.Series([]), lcos=lcos, n_dispatch_prev=0)
        elif scenario == "bat" and start_hour != start_hour_0:
            n_redispatch = solve_redispatch_network_with_bat(n_24, n_dispatch, network_year,
                                                             start_hour=start_hour, start_hour_0=start_hour_0,
                                                             c_rate=c_rate, flex_potential=flex_potential,
                                                             plant_potential=plant_potential, storage_ops=storage_ops,
                                                             s_SOC_batteries=s_SOC_batteries, lcos=lcos, n_dispatch_prev=n_dispatch_prev)

        # Initialize series for passing the 24th snapshot SOC value to next day network
        # ---------------
        if scenario == "bat":
            s_SOC_batteries = n_redispatch.stores_t.e.iloc[-1].copy()

        # Append networks to corresponding yearly lists
        # ---------------
        l_networks_24.append(n_24)
        l_networks_dispatch.append(n_dispatch)
        l_networks_redispatch.append(n_redispatch)

        # save dispatch network for next day
        n_dispatch_prev = n_dispatch.copy()


    # For each results list: Create a network out of daily networks for dispatch & redispatch
    # ---------------
    network_dispatch = concat_network(l_networks_dispatch)
    network_redispatch = concat_network(l_networks_redispatch)

    return network_dispatch, network_redispatch


def solve_redispatch_workflow_future(filename, year, c_rate=0.25):
    """
    Function to run the redispatch workflow for one network in the networks_redispatch folder.
    Used to compare bat and no bat scenario for a single network.
    """

    folder = r'/cluster/home/wlaumen/Euler/pypsa-eur/networks_redispatch'
    #folder = r'C:\Users\Willem\pypsa-eur\networks_redispatch'
    filename = filename

    path_n = folder + "/" + filename + ".nc"
    print(path_n)
    # Define network
    n = pypsa.Network(path_n)
    n_future = n.copy()
    print(n_future.generators.groupby("carrier").sum())

    # Scenario parameters
    if year == 2025:
        flex_potential = 3200
        plant_potential = 1200
    elif year == 2030:
        flex_potential = 8000
        plant_potential = 2000


    # Run redispatch with batteries & export files
    export_path = folder + r"/results"


    # n_d, n_rd = redispatch_workflow_future(n_future = n_future, c_rate=c_rate, storage_ops="none", flex_potential=flex_potential,
    #                                        plant_potential=plant_potential, scenario="no bat", ratio_wind=2.2, ratio_pv=1.38, lcos=0)
    # # export solved dispatch & redispatch workflow as well as objective value list
    # n_d.export_to_netcdf(path=export_path + r"/dispatch/" + filename + ".nc", export_standard_types=False, least_significant_digit=None)
    # n_rd.export_to_netcdf(path=export_path + r"/redispatch/" + filename + ".nc", export_standard_types=False, least_significant_digit=None)
    # gc.collect()
    #
    # del n_future, n
    # n = pypsa.Network(path_n)
    # n_future = n.copy()
    #
    # # Redispatch batteries LOAD
    #
    #
    # # Redispatch batteries supply
    # n_d_bat_supply, n_rd_bat_supply = redispatch_workflow_future(n_future = n_future, c_rate=c_rate, storage_ops="supply", flex_potential=flex_potential,
    #                                                 plant_potential=plant_potential, scenario="bat", ratio_wind=2.2, ratio_pv=1.38, lcos=0)
    # print("\n\nSave bat_supply_to_nc\n\n")
    # n_rd_bat_supply.export_to_netcdf(path=export_path + r"/redispatch/" + filename + "_bat_supply.nc", export_standard_types=False, least_significant_digit=None)
    # gc.collect()
    #
    # del n_future, n
    # n = pypsa.Network(path_n)
    # n_future = n.copy()
    #
    # # Redispatch batteries all lcos
    # n_d_bat_all, n_rd_bat_all = redispatch_workflow_future(n_future = n_future, c_rate=c_rate, storage_ops="all", flex_potential=flex_potential,
    #                                                 plant_potential=plant_potential, scenario="bat", ratio_wind=2.2, ratio_pv=1.38, lcos=0)
    # print("\n\nSave bat_all_to_nc\n\n")
    # n_rd_bat_all.export_to_netcdf(path=export_path + r"/redispatch/" + filename + "_bat_all.nc", export_standard_types=False, least_significant_digit=None)
    # gc.collect()





    # n_d_bat_load, n_rd_bat_load = redispatch_workflow_future(n_future = n_future, c_rate=c_rate, storage_ops="load", flex_potential=flex_potential,
    #                                                   plant_potential=plant_potential, scenario="bat", ratio_wind=2.2, ratio_pv=1.38, lcos=0)
    # print("\n\nSave bat_load_to_nc\n\n")
    # n_rd_bat_load.export_to_netcdf(path=export_path + r"/redispatch/" + filename + "_bat_load.nc", export_standard_types=False, least_significant_digit=None)
    # gc.collect()
    #
    # del n_future, n
    # n = pypsa.Network(path_n)
    # n_future = n.copy()

    #############
    # WITH LCOS #
    #############

    # Redispatch batteries LOAD
    # n_d_bat_load_lcos, n_rd_bat_load_lcos = redispatch_workflow_future(n_future=n_future, c_rate=c_rate, storage_ops="load",flex_potential=flex_potential,
    #                                                          plant_potential=plant_potential, scenario="bat", ratio_wind=2.2, ratio_pv=1.38, lcos=46.43)
    # print("\n\nSave bat_load_to_nc\n\n")
    # n_rd_bat_load_lcos.export_to_netcdf(path=export_path + r"/redispatch/" + filename + "_bat_load_lcos.nc",
    #                                export_standard_types=False, least_significant_digit=None)
    # gc.collect()
    #
    # del n_future, n
    # n = pypsa.Network(path_n)
    # n_future = n.copy()

    # # Redispatch batteries Supply
    # n_d_bat_supply_lcos, n_rd_bat_supply_lcos = redispatch_workflow_future(n_future=n_future, c_rate=c_rate, storage_ops="supply", flex_potential=flex_potential,
    #                                                                        plant_potential=plant_potential, scenario="bat", ratio_wind=2.2, ratio_pv=1.38, lcos=46.43)
    # print("\n\nSave bat_supply_to_nc\n\n")
    # n_rd_bat_supply_lcos.export_to_netcdf(path=export_path + r"/redispatch/" + filename + "_bat_supply_lcos.nc",
    #                                       export_standard_types=False, least_significant_digit=None)
    # gc.collect()

    # del n_future, n
    # n = pypsa.Network(path_n)
    # n_future = n.copy()

    # # Redispatch batteries Supply
    # n_d_bat_all_lcos, n_rd_bat_all_lcos = redispatch_workflow_future(n_future=n_future, c_rate=c_rate, storage_ops="all", flex_potential=flex_potential,
    #                                                                  plant_potential=plant_potential, scenario="bat", ratio_wind=2.2, ratio_pv=1.38, lcos=46.43)
    # print("\n\nSave bat_all_to_nc\n\n")
    # n_rd_bat_all_lcos.export_to_netcdf(path=export_path + r"/redispatch/" + filename + "_bat_all_lcos.nc", export_standard_types=False, least_significant_digit=None)
    # gc.collect()


    # Quarterly year simulations
    # ##########################
    # Redispatch batteries Supply

    n_d_6570, n_rd_6570 = redispatch_workflow_future(n_future=n_future, c_rate=c_rate, storage_ops="none", flex_potential=flex_potential,
                                                                     plant_potential=plant_potential, scenario="no bat", ratio_wind=2.2, ratio_pv=1.38, lcos=0)
    print("\n\nSave bat_all_to_nc\n\n")
    n_d_6570.export_to_netcdf(path=export_path + r"/dispatch/" + filename + "_6570.nc", export_standard_types=False, least_significant_digit=None)
    n_rd_6570.export_to_netcdf(path=export_path + r"/redispatch/" + filename + "_6570.nc", export_standard_types=False, least_significant_digit=None)
    gc.collect()

    del n_future, n
    n = pypsa.Network(path_n)
    n_future = n.copy()


    n_d_bat_all_6570, n_rd_bat_all_6570 = redispatch_workflow_future(n_future=n_future, c_rate=c_rate, storage_ops="all", flex_potential=flex_potential,
                                                                     plant_potential=plant_potential, scenario="bat", ratio_wind=2.2, ratio_pv=1.38, lcos=0)
    print("\n\nSave bat_all_to_nc\n\n")
    n_rd_bat_all_6570.export_to_netcdf(path=export_path + r"/redispatch/" + filename + "_bat_all_6570.nc", export_standard_types=False, least_significant_digit=None)
    gc.collect()

    del n_future, n
    n = pypsa.Network(path_n)
    n_future = n.copy()


    n_d_bat_all_lcos_6570, n_rd_bat_all_lcos_6570 = redispatch_workflow_future(n_future=n_future, c_rate=c_rate, storage_ops="all", flex_potential=flex_potential,
                                                                     plant_potential=plant_potential, scenario="bat", ratio_wind=2.2, ratio_pv=1.38, lcos=46.43)
    print("\n\nSave bat_all_to_nc\n\n")
    n_rd_bat_all_lcos_6570.export_to_netcdf(path=export_path + r"/redispatch/" + filename + "_bat_all_lcos_6570.nc", export_standard_types=False, least_significant_digit=None)
    gc.collect()

    del n_future, n
    n = pypsa.Network(path_n)
    n_future = n.copy()


    n_d_bat_all_lcos_20, n_rd_bat_all_lcos_20 = redispatch_workflow_future(n_future=n_future, c_rate=c_rate, storage_ops="all", flex_potential=flex_potential,
                                                                     plant_potential=plant_potential, scenario="bat", ratio_wind=2.2, ratio_pv=1.38, lcos=33.87)
    print("\n\nSave bat_all_to_nc\n\n")
    n_rd_bat_all_lcos_20.export_to_netcdf(path=export_path + r"/redispatch/" + filename + "_bat_all_lcos_20%.nc", export_standard_types=False, least_significant_digit=None)
    gc.collect()




def main():
    solve_redispatch_workflow_future(filename = "elec_s300_220_ec_lcopt_1H-Ep-noex_2025_future", year=2025, c_rate=0.25)
if __name__ == "__main__":
    main()