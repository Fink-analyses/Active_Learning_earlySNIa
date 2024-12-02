import os
import numpy as np
import pandas as pd
from actsnfink import *
import matplotlib as mpl
import matplotlib.pylab as plt


fink_colors_list = ["#15284F", "#F5622E", "#D5D5D3", "#3C8DFF"]
# Colors to plot
colordic = {1: "C0", 2: "C1", "g": "C0", "r": "C1"}

# Labels of ZTF filters
filtdic = {1: "g", 2: "r"}


mpl.rcParams["font.size"] = 16
mpl.rcParams["legend.fontsize"] = "medium"
mpl.rcParams["figure.titlesize"] = "large"
mpl.rcParams["lines.linewidth"] = 3


def plot_lc_mag(lc, proba, dir_suffix=""):
    """Plot and save light-curve

    Args:
        lc (pd.DataFrame): light-curve data
        proba (float): classification probability
        dir_suffix (str): suffix for outpath
    """
    fig = plt.figure(figsize=(10, 5))

    for filt in filtdic.keys():
        jd = lc.cjd[0][np.where(lc.cfid[0] == filt)[0]] - 2400000.5
        mag = lc.cmagpsf[0][np.where(lc.cfid[0] == filt)[0]]
        emag = lc.csigmapsf[0][np.where(lc.cfid[0] == filt)[0]]
        plt.errorbar(
            jd, mag, emag, ls="", marker="o", color=colordic[filt], label=filtdic[filt]
        )

    plt.gca().invert_yaxis()
    lcid = lc["objectId"].values[0]
    lctype = lc["TNS"].values[0]
    plt.title(f"{lcid} Type: {lctype} Proba:{np.round(proba,2)}")
    plt.xlabel("Modified Julian Date")
    plt.ylabel("Magnitude")

    outdir = (
        f"../plots/train_lcs_{dir_suffix}" if dir_suffix != "" else "../plots/train_lcs"
    )
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{lcid}.png")

    plt.close()


def plot_lc_flux_wfit(lc, proba, alerts_features, dir_suffix=""):
    """Plot lc in mag and flux space with fitted sigmoid

    Args:
        lc (pd.DataFrame): light-curve data in FLUX space
        proba (float): classification probability
        alerts_features (pd.DataFrame): alerts features (sigmoid)
        dir_suffix (str): suffix for outpath
    """
    fig = plt.figure(figsize=(10, 5))

    def sigmoid(t, c, a, b):
        return c / (1 + np.exp(-a * (t - b)))

    JD_fitted = {}
    FLUX_fitted = {}

    for filt in lc["FLT"].unique():
        sel_flt = lc[lc["FLT"] == filt]
        jd = sel_flt.MJD.values - 2400000.5
        flux = sel_flt.FLUXCAL.values
        eflux = sel_flt.FLUXCALERR.values
        plt.errorbar(
            jd,
            flux,
            eflux,
            ls="",
            marker="o",
            color=colordic[filt],
            label=filt,
        )
        # plto sigmoid
        jd_fitted2 = np.linspace(0, max(jd) - min(jd), 1000)
        flux_fitted = get_predicted_flux(
            jd_fitted2,
            alerts_features[f"a_{filt}"].values[0],
            alerts_features[f"b_{filt}"].values[0],
            alerts_features[f"c_{filt}"].values[0],
        )
        plt.plot(jd_fitted2 + min(jd), flux_fitted, color=colordic[filt])

    lcid = lc["id"].values[0]
    lctype = lc["type"].values[0]
    plt.title(f"{lcid} Type: {lctype} Proba:{np.round(proba,2)}")
    plt.xlabel("Modified Julian Date")
    plt.ylabel("FLUXCAL")

    outdir = (
        f"../plots/train_lcs_{dir_suffix}" if dir_suffix != "" else "../plots/train_lcs"
    )
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{lcid}_wfit.png")

    plt.close()


def plot_metrics_listdf(
    list_df, label_list, plots_dir="./", varx="date_universal", suffix=""
):
    """Plot metrics

    Args:
        list_df (list): List of pd.DataFrames with metrics values
        label_list (list): list of strings with labels to use in legend
        plots_dir (str, optional): outdir. Defaults to './'.
        varx (str, optional): x variable. Defaults to 'date'.
        suffix (str, optional): suffix AL loop type. Defaults to ''.
    """
    os.makedirs(plots_dir, exist_ok=True)
    # Reformat date if needed
    if varx == "date_universal":

        all_dates = np.concat([df["date"].values for df in list_df])
        all_dates.sort()
        to_merge = pd.DataFrame(
            {"date": all_dates, "date_universal": np.arange(1, len(all_dates) + 1)}
        )

        new_list_df = []
        for df in list_df:
            new_list_df.append(pd.merge(df, to_merge, on="date", how="left"))

    else:
        new_list_df = list_df

    plt.figure(figsize=(16, 10), tight_layout=True)

    for i, df in enumerate(new_list_df):

        color_to_use = fink_colors_list[i]
        xlabel = "normalised date" if varx == "date_universal" else varx

        plt.subplot(2, 2, 1)
        plt.scatter(
            df[varx].astype(int),
            df["accuracy"],
            label=label_list[i],
            color=color_to_use,
        )
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel("accuracy")

        plt.subplot(2, 2, 2)
        plt.scatter(df[varx].astype(int), df["efficiency"], color=color_to_use)
        plt.xlabel(xlabel)
        plt.ylabel("efficiency")

        plt.subplot(2, 2, 3)
        plt.scatter(df[varx].astype(int), df["purity"], color=color_to_use)
        plt.xlabel(xlabel)
        plt.ylabel("purity")

        plt.subplot(2, 2, 4)
        plt.scatter(df[varx].astype(int), df["fom"], color=color_to_use)
        plt.xlabel(xlabel)
        plt.ylabel("figure of merit")

    plt.savefig(f"{plots_dir}/metrics_superposed_{varx}{suffix}.png")
    plt.show()
