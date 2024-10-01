import os
import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta

observing_periods = {1:[Time('2023-09-25', format='iso').jd,Time('2023-10-23', format='iso').jd], 
                     2:[Time('2024-02-25', format='iso').jd,Time('2024-02-29', format='iso').jd], 
                     3:[Time('2024-04-04', format='iso').jd,Time('2024-06-02', format='iso').jd],
                     4:[Time('2024-06-24', format='iso').jd,Time('2024-08-15', format='iso').jd]}


def add_days_and_format(x,days):
    """Function to convert Time object to desired format

    Args:
        x (str): Date in tring format
        days (int): number of days to add

    Returns:
        str: Date plus delta
    """
    # Convert Time object to datetime
    time_obj = Time(x, format='iso') + TimeDelta(days, format='jd')
    dt = time_obj.datetime 
    # dt = time_obj.datetime
    # Format datetime to custom format
    return f"{dt.year}{dt.month:02d}{dt.day:02d}"

def extract_ztf_names(row):
    """Function to extract names starting with 'ZTF'

    Args:
        row (pd.row):  DataFrame row

    Returns:
        str: ZTF name
    """
    if row:
        ztf_list = [name.strip() for name in row.split(',') if name.strip().startswith('ZTF')]
        if len(ztf_list)>0:
            return ztf_list[0]
        else:
            return ""
    else:
        return ""
    
def load_TNS_Fink(fname):
    """Read Fink TNS file and format information

    Args:
        fname (str): TNS filename (Fink .parquet)

    Returns:
        pd.DataFrame: DataFrame with TNS classified ZTF detected events
    """
    # read TNS labels
    df_tns_tmp = pd.read_parquet(fname)

    # constrain date
    df_tns_tmp['discoveryjd'] = [Time(d, format='iso').jd for d in df_tns_tmp['discoverydate']]
    df_tns = df_tns_tmp[df_tns_tmp['discoveryjd']> Time('2023-09-21', format='iso').jd].copy()

    # OBSERVING PERIODS THIS CAMPAIGN
    df_tns['discoveryjd+12'] = df_tns['discoveryjd'].apply(lambda x: int(x+12)) # 12 dayS as time to get ~3 measurements in each band
    mask1 = (df_tns['discoveryjd+12']>observing_periods[1][0]) & (df_tns['discoveryjd+12']<observing_periods[1][1])
    mask2 = (df_tns['discoveryjd+12']>observing_periods[2][0]) & (df_tns['discoveryjd+12']<observing_periods[2][1])
    mask3 = (df_tns['discoveryjd+12']>observing_periods[3][0]) & (df_tns['discoveryjd+12']<observing_periods[3][1])
    mask4 = (df_tns['discoveryjd+12']>observing_periods[4][0]) & (df_tns['discoveryjd+12']<observing_periods[4][1])
    df_tns_indaterange = df_tns[mask1 | mask2 | mask3 | mask4]

    # with ZTF data
    tns_class_ztf_lc = df_tns_indaterange[df_tns_indaterange['internal_names'].str.contains('ZTF')].copy()
    tns_class_ztf_lc.loc[:,'ztf_names'] = tns_class_ztf_lc.loc[:,'internal_names'].apply(extract_ztf_names).to_numpy()

    # reformat
    tns_class_ztf_lc.loc[:,'type'] = tns_class_ztf_lc.loc[:,'type'].str.strip('(TNS) SN ')
    tns_class_ztf_lc['type AL'] = tns_class_ztf_lc.loc[:,'type'].apply(lambda x: 'Ia' if 'Ia' in x else 'other')
    tns_class_ztf_lc['discoveryjd+12_strfmt'] = tns_class_ztf_lc.loc[:,'discoverydate'].apply(lambda x: add_days_and_format(x, 12))
    tns_class_ztf_lc.loc[:,'internal_names'] = tns_class_ztf_lc.loc[:,'internal_names'].fillna(value="")

    tns_class_ztf_lc = tns_class_ztf_lc[['ztf_names','type AL','discoveryjd+12','discoveryjd+12_strfmt','type','reporting_group']].copy()
    # sort
    tns_class_ztf_lc = tns_class_ztf_lc.sort_values(by='discoveryjd+12')


    return tns_class_ztf_lc