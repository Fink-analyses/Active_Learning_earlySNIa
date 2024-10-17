import os
import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta

observing_periods = {
    1: [Time("2023-09-25", format="iso").jd, Time("2023-10-23", format="iso").jd],
    2: [Time("2024-02-25", format="iso").jd, Time("2024-02-29", format="iso").jd],
    3: [Time("2024-04-04", format="iso").jd, Time("2024-06-02", format="iso").jd],
    4: [Time("2024-06-24", format="iso").jd, Time("2024-08-15", format="iso").jd],
}


def add_days_and_format(x, days):
    """Function to convert Time object to desired format

    Args:
        x (str): Date in tring format
        days (int): number of days to add

    Returns:
        str: Date plus delta
    """
    # Convert Time object to datetime
    time_obj = Time(x, format="iso") + TimeDelta(days, format="jd")
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
        ztf_list = [
            name.strip() for name in row.split(",") if name.strip().startswith("ZTF")
        ]
        if len(ztf_list) > 0:
            return ztf_list[0]
        else:
            return ""
    else:
        return ""


def load_TNS_Fink(fname, ndaysplus=9):
    """Read Fink TNS file and format information

    Args:
        fname (str): TNS filename (Fink .parquet)

    Returns:
        pd.DataFrame: DataFrame with TNS classified ZTF detected events
    """
    # read TNS labels
    df_tmp = pd.read_parquet(fname)

    # with ZTF data
    df_tmp.loc[:, "internal_names"] = df_tmp.loc[:, "internal_names"].fillna(value="")
    df = df_tmp[df_tmp["internal_names"].str.contains("ZTF")].copy()
    df.loc[:, "ztf_names"] = (
        df.loc[:, "internal_names"].apply(extract_ztf_names).to_numpy()
    )

    # reformatting time columns
    df["discoveryjd"] = [Time(d, format="iso").jd for d in df["discoverydate"]]
    df[f"discoveryjd+{ndaysplus}"] = df["discoveryjd"].apply(
        lambda x: int(x + ndaysplus)
    )  # time to get ~3 measurements in each band
    df[f"discoveryjd+{ndaysplus}_strfmt"] = df.loc[:, "discoverydate"].apply(
        lambda x: add_days_and_format(x, ndaysplus)
    )

    # # constraining observing periods
    mask1 = (df[f"discoveryjd+{ndaysplus}"] > observing_periods[1][0]) & (
        df[f"discoveryjd+{ndaysplus}"] < observing_periods[1][1]
    )
    mask2 = (df[f"discoveryjd+{ndaysplus}"] > observing_periods[2][0]) & (
        df[f"discoveryjd+{ndaysplus}"] < observing_periods[2][1]
    )
    mask3 = (df[f"discoveryjd+{ndaysplus}"] > observing_periods[3][0]) & (
        df[f"discoveryjd+{ndaysplus}"] < observing_periods[3][1]
    )
    mask4 = (df[f"discoveryjd+{ndaysplus}"] > observing_periods[4][0]) & (
        df[f"discoveryjd+{ndaysplus}"] < observing_periods[4][1]
    )
    df_indaterange = df[mask1 | mask2 | mask3 | mask4]

    # reformat
    df_indaterange.loc[:, "type"] = df_indaterange.loc[:, "type"].str.strip("(TNS) SN ")
    df_indaterange["type AL"] = df_indaterange.loc[:, "type"].apply(
        lambda x: "Ia" if "Ia" in x else "other"
    )

    df_out = df_indaterange[
        [
            "ztf_names",
            "type AL",
            f"discoveryjd+{ndaysplus}",
            f"discoveryjd+{ndaysplus}_strfmt",
            "type",
            "reporting_group",
        ]
    ].copy()
    # sort
    df_out = df_out.sort_values(by=f"discoveryjd+{ndaysplus}")

    return df_out
