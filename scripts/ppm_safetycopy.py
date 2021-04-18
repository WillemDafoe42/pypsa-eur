#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 23:02:46 2021

@author: chiara
"""

import powerplantmatching as pm
import pypsa
import pandas as pd
from powerplantmatching.export import map_country_bus
#
n = pypsa.Network(r'C:\Users\Willem\pypsa-eur\networks\elec_s300_200_ec_lcopt_1H-noex.nc')
#
available = ['DE', 'FR', 'PL', 'CH', 'DK', 'CZ', 'SE', 'GB']
tech_map = {'Onshore': 'onwind', 'Offshore': 'offwind', 'Solar': 'solar'}
countries = set(available) & set(n.buses.country)
techs = ['onwind','offwind','solar']
tech_map = {k: v for k, v in tech_map.items() if v in techs}
#
df = pd.concat([pm.data.OPSD_VRE_country(c) for c in countries])
technology_b = ~df.Technology.isin(['Onshore', 'Offshore'])
df['Fueltype'] = df.Fueltype.where(technology_b, df.Technology)
df = df.query('Fueltype in @tech_map').powerplant.convert_country_to_alpha2()
#
caps = {tc: pd.DataFrame for tc in techs}
#
for fueltype, carrier_like in tech_map.items():
    gens = n.generators[lambda df: df.carrier.str.contains(carrier_like)]
    buses = n.buses.loc[gens.bus.unique()]    
    gens_per_bus = gens.groupby('bus').p_nom.count()
    caps[carrier_like] = map_country_bus(df.query('Fueltype == @fueltype'), buses)
    caps[carrier_like] = caps[carrier_like].groupby(['bus']).Capacity.sum()
    caps[carrier_like] = caps[carrier_like] / gens_per_bus.reindex(caps[carrier_like].index, fill_value=1)
    print(fueltype)
    print(caps[carrier_like].sum() / 1000)
    n.generators.p_nom.update(gens.bus.map(caps[carrier_like]).dropna())
#
print(n.generators.p_nom.filter(like="onwind").sum() / 1000)
print(n.generators.p_nom.filter(like="offwind").sum() / 1000)
print(n.generators.p_nom.filter(like="solar").sum() / 1000)
#
vergleich = {tcs: pd.DataFrame() for tcs in techs}
for tcs in techs:
    for i in caps[tcs].index:
        try:
            ll = n.generators[n.generators.carrier==tcs].set_index('bus')
            vergleich[tcs].loc[i,'caps'] = caps[tcs][i] / 1000
            vergleich[tcs].loc[i,'network'] = ll.loc[i,'p_nom'] / 1000
            vergleich[tcs].loc[i,'differenz'] = vergleich[tcs].loc[i,'caps'] - vergleich[tcs].loc[i,'network']
        except:
            pass
#
for tcs in techs:            
    print(vergleich[tcs].sum()) 
#
n.export_to_netcdf(path=r'C:\Users\Willem\pypsa-eur\networks\elec_s300_200_ec_lcopt_1H-noex_new.nc')
