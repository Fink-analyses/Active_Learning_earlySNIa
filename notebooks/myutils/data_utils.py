import os
import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta


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


def load_TNS_reformat(fname):
    """Read Fink TNS file and format information

    Args:
        fname (str): TNS filename (Fink .parquet)

    Returns:
        pd.DataFrame: DataFrame with reformatted type for AL loop
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
    df["discoveryjd"] = [int(Time(d, format="iso").jd) for d in df["discoverydate"]]

    # reformat
    df.loc[:, "type"] = df.loc[:, "type"].str.strip("(TNS) SN ")
    df["type AL"] = df.loc[:, "type"].apply(lambda x: "Ia" if "Ia" in x else "other")

    df_out = df[
        [
            "ztf_names",
            "type AL",
            "discoveryjd",
            "type",
            "reporting_group",
        ]
    ].copy()
    # sort
    df_out = df_out.sort_values(by=f"discoveryjd")

    return df_out


def apply_timerange_fink_fup(df, ndaysplus=9):
    """Apply AL follow-up time range given a time after discovery

    Args:
        df ([pd.DatatFrame]): reformatted TNS dataframe
        ndaysplus (int, optional): How many days after discovery this analysis uses. Defaults to 9.

    Returns:
        [pd.DatatFrame]: TNS dataframe with time range of AL SSO follow-up
    """

    observing_periods = {
        1: [Time("2023-09-25", format="iso").jd, Time("2023-10-23", format="iso").jd],
        2: [Time("2024-02-25", format="iso").jd, Time("2024-02-29", format="iso").jd],
        3: [Time("2024-04-04", format="iso").jd, Time("2024-06-02", format="iso").jd],
        4: [Time("2024-06-24", format="iso").jd, Time("2024-08-15", format="iso").jd],
    }

    df["fup requested"] = df["discoveryjd"].apply(lambda x: int(x + ndaysplus))
    df["fup requested (str)"] = df.loc[:, "fup requested"].apply(
        lambda x: Time(x, format="jd").strftime("%Y%m%d")
    )

    # 1 day more for label acquisition
    df["label acquired"] = df["discoveryjd"].apply(lambda x: int(x + ndaysplus + 1))

    # constraining observing periods
    mask1 = (df["label acquired"] > observing_periods[1][0]) & (
        df["label acquired"] < observing_periods[1][1]
    )
    mask2 = (df["label acquired"] > observing_periods[2][0]) & (
        df["label acquired"] < observing_periods[2][1]
    )
    mask3 = (df["label acquired"] > observing_periods[3][0]) & (
        df["label acquired"] < observing_periods[3][1]
    )
    mask4 = (df["label acquired"] > observing_periods[4][0]) & (
        df["label acquired"] < observing_periods[4][1]
    )
    df_indaterange = df[mask1 | mask2 | mask3 | mask4]

    return df_indaterange
