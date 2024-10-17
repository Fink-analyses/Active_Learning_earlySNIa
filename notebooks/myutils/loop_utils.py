import os, io
import shutil
import pickle
import requests
import numpy as np
import pandas as pd
from actsnfink import *
from copy import deepcopy
from astropy.time import Time
from . import plot_utils as pu
from actsnclass.metrics import get_snpcc_metric
from sklearn.ensemble import RandomForestClassifier


def train_and_test_initial_model(train_for_loop, test_for_loop, strategy):
    """Train and test initial model in loop

    Args:
        train_for_loop (pd.DataFrame): Training data
        test_for_loop (pd.DataFrame): Testing data
        strategy (str): AL strategy used
    """
    # Stats train
    print(f"Number of training light-curves {len(train_for_loop)}")
    train_for_loop.groupby("type").count()["c_g"]

    # train initial model
    clf = RandomForestClassifier(random_state=42, n_estimators=1000)
    clf.fit(
        train_for_loop[list(train_for_loop.keys())[:-3]],
        train_for_loop["type"].values == "Ia",
    )
    # save mode
    pickle.dump(
        clf, open("../data/initial_state/" + strategy + "/models/model.pkl", "wb")
    )

    # make predictions
    pred = clf.predict(test_for_loop[list(test_for_loop.keys())[:-2]])

    # calculate metrics
    metric_names, metric_values = get_snpcc_metric(
        pred, test_for_loop["type"].values == "Ia"
    )

    output = [f"{k} {round(metric_values[i],2)}" for i, k in enumerate(metric_names)]
    print(output)


def create_directory_structure(dir_output, date, strategy):
    """Create directory structure

    Args:
        date (str): date of loop
    """
    dirname_output = dir_output + date + "/"

    for name in [
        dirname_output + "/",
        dirname_output + "/" + strategy + "/",
        dirname_output + "/" + strategy + "/models/",
        dirname_output + "/" + strategy + "/class_prob/",
        dirname_output + "/" + strategy + "/metrics/",
        dirname_output + "/" + strategy + "/queries/",
        dirname_output + "/" + strategy + "/training_samples/",
        dirname_output + "/" + strategy + "/test_samples/",
    ]:
        os.makedirs(name)


def read_previous_state(d, date, new_labels, dir_output, strategy, verbose=False):
    """Load RF model and predictions from previous date

    Args:
        d (int): counter on loop
        date (str): date in stirng format
        new_labels (dict): labels in the AL loop
        dir_output (str): output directory

    Returns:
        pd.DataFrame, RFmodel, pd.DataFrame: training data, RF model, training probabilities
    """

    def read_initial_state(dir_output, strategy):
        """Read initial state

        Args:
            strategy (str): AL strategy used

        Returns:
            pd.DataFrame, RFmodel, pd.DataFrame: training data, RF model, training probabilities
        """
        print("Read initial state")
        train_for_loop = pd.read_csv(
            dir_output
            + "initial_state/"
            + strategy
            + "/training_samples/train_after_loop.csv",
            index_col=False,
        )

        train_for_loop_probas = pd.DataFrame()
        train_for_loop_probas["objectId"] = train_for_loop["objectId"].values
        train_for_loop_probas["probability"] = np.zeros(len(train_for_loop))

        clf_before = pickle.load(
            open(dir_output + "initial_state/" + strategy + "/models/model.pkl", "rb")
        )

        return train_for_loop, clf_before, train_for_loop_probas

    # first date - > initial state
    if date == list(new_labels.keys())[0]:
        train_for_loop, clf_before, train_for_loop_probas = read_initial_state(
            dir_output, strategy
        )

    # not first date -> check previous dates until find a file -> if not, use initial state
    else:
        # loop over previous dates
        is_file = False
        indx = list(new_labels.keys()).index(date) - 1
        while not is_file and indx >= 0:
            date_before = list(new_labels.keys())[indx]
            dir_before = dir_output + date_before + "/" + strategy
            is_file = os.path.isfile(dir_before + "/training_samples/train.csv")
            if is_file:
                train_for_loop = pd.read_csv(
                    dir_before + "/training_samples/train.csv",
                    index_col=False,
                )

                clf_before = pickle.load(
                    open(
                        dir_before + "/models/model.pkl",
                        "rb",
                    )
                )
                train_for_loop_probas = pd.read_csv(
                    dir_before + "/training_samples/probabilities.csv",
                    index_col=False,
                )
                print(f"Read {date_before} state")
            else:
                indx = indx - 1

        # if no previous state, use initial state
        if not is_file:
            print("No previous date found")
            train_for_loop, clf_before, train_for_loop_probas = read_initial_state(
                dir_output, strategy
            )

    return train_for_loop, clf_before, train_for_loop_probas


def get_alert_data(new_labels, date, verbose=False):
    """Get Fink alert data

    Args:
        new_labels (dic): dictionary with AL loop labels
        date (str): date currently in the loop
        verbose (bool, optional): If print information. Defaults to False.

    Returns:
        list: list containing all alert information
    """
    alerts_list = []

    # loop over all objects in the date
    for i in range(new_labels[date].shape[0]):

        name = new_labels[date][i][0]

        r = requests.post(
            "https://fink-portal.org/api/v1/objects",
            json={
                "objectId": name,
                "output-format": "json",
                "withupperlim": "True",  # online ML uses bad quality
            },
        )

        try:
            # Format output in a DataFrame
            pdf = pd.read_json(io.BytesIO(r.content))
            pdf = pdf[
                pdf["d:tag"] != "upperlim"
            ]  # using valida and badquality detections
        except Exception:
            # empty DataFrame
            print(f"Empty data for {name} or error in request {r}")
            pdf = pd.DataFrame()

        if not pdf.shape[0] == 0:
            # add label
            pdf["type"] = new_labels[date][i][1]
            if verbose:
                print(f"Got data for {name}")

            alerts_list.append(pdf)

    return alerts_list


def AL_loop(
    new_labels,
    strategy,
    test_for_loop,
    dir_suffix="",
    output_path="../dump/",
    proba_cut=True,
    plot_lcs=False,
    verbose=False,
):
    """Active Learning loop

    Args:
        new_labels (Dict(np.array)): Dictionary with dates key (str) and nup. array with [name, label, date]
        strategy (str): AL strategy used
        test_for_loop (pd.DataFrame): Features of test set
        dir_suffix (str, optional): output directory suffix. Defaults to "".
        proba_cut (bool, optional): select only alerts within a range of probabilities. Defaults to False.
        plot_lcs (bool, optional): Plot light-curves used in the loop. Defaults to False.
        verbose (bool, optional): Verbose. Defaults to False.

    Returns:
        pd.DataFrame: metrics for all dates
    """

    # create output directory structure (removing previous)
    dir_output = (
        f"{output_path}/data_{dir_suffix}/"
        if dir_suffix != ""
        else f"{output_path}/data/"
    )
    if os.path.exists(dir_output):
        shutil.rmtree(dir_output)
    # copy initial state
    os.makedirs(f"{dir_output}", exist_ok=True)
    shutil.copytree(
        "../data/initial_state",
        f"{dir_output}/initial_state/",
        ignore=shutil.ignore_patterns(".DS_Store"),
    )

    metrics_all_list = []
    for d, date in enumerate(list(new_labels.keys())):

        if verbose:
            print(date)

        # read previous state
        train_for_loop, clf_before, train_for_loop_probas = read_previous_state(
            d, date, new_labels, dir_output, strategy, verbose=verbose
        )

        # Get alert data until follow-up date
        alerts_list = get_alert_data(new_labels, date, verbose=False)

        # Check data from Fink
        if len(alerts_list) > 0:
            # create output directory structure
            create_directory_structure(dir_output, date, strategy)

            # reformat alerts into a list
            alerts_format_list = []
            alerts_pd = pd.concat(alerts_list, ignore_index=True)

            for indx_obj in range(new_labels[date].shape[0]):
                # isolate one object
                flag_obj = (
                    alerts_pd["i:objectId"].values == new_labels[date][indx_obj][0]
                )

                sel_obj = alerts_pd[flag_obj]
                sel_obj_wdaterange = sel_obj[
                    sel_obj["i:jd"] < float(new_labels[date][indx_obj][2])
                ]
                if len(sel_obj_wdaterange) > 0:
                    # check if we have data
                    candid = sel_obj_wdaterange["i:candid"].values[
                        np.argsort(sel_obj_wdaterange["i:jd"].values)[-1]
                    ]
                    lc = pd.DataFrame(
                        [
                            [
                                new_labels[date][indx_obj][0],
                                candid,
                                sel_obj_wdaterange["i:jd"].values,
                                sel_obj_wdaterange["i:fid"].values,
                                sel_obj_wdaterange["i:magpsf"].values,
                                sel_obj_wdaterange["i:sigmapsf"].values,
                                new_labels[date][indx_obj][1],
                            ]
                        ],
                        columns=[
                            "objectId",
                            "candid",
                            "cjd",
                            "cfid",
                            "cmagpsf",
                            "csigmapsf",
                            "TNS",
                        ],
                        index=[0],
                        dtype=object,
                    )
                    alerts_format_list.append(lc)

            # Add data to the loop
            if len(alerts_format_list) > 0:
                alerts_format_pd = pd.concat(alerts_format_list, ignore_index=True)

                # convert from mag to FLUXCAL
                alerts_flux = convert_full_dataset(
                    alerts_format_pd, obj_id_header="objectId"
                )

                # extract features
                alerts_features = featurize_full_dataset(
                    alerts_flux,
                    screen=False,
                    ewma_window=3,
                    min_rising_points=1,
                    min_data_points=3,
                    rising_criteria="ewma",
                )

                # filter alerts with zero in all filters
                flag_zero = np.logical_and(
                    alerts_features["a_g"].values == 0.0,
                    alerts_features["a_r"].values == 0.0,
                )
                # flag_zero = np.logical_or(alerts_features['a_g'].values == 0.0,
                #                            alerts_features['a_r'].values == 0.0)
                alerts_use = deepcopy(alerts_features[~flag_zero])

                if verbose:
                    if len(alerts_features[flag_zero]) > 0:
                        print(f'Missing {alerts_features[flag_zero]["id"]}')
                        print(
                            alerts_flux[
                                alerts_flux["id"].isin(alerts_features[flag_zero]["id"])
                            ]
                        )

                # go through events if they have at least one band with features
                if alerts_use.shape[0] > 0:

                    alerts_use.rename(columns={"id": "objectId"}, inplace=True)
                    alerts_use["loop"] = (
                        list(new_labels.keys()).index(date) + 30
                    )  # probably wrong to put 30 but to recheck (dont think is used)

                    # see what current model says about queried alerts
                    pred_prob_query = clf_before.predict_proba(
                        alerts_use[list(alerts_use.keys())[2:-1]]
                    )

                    # Make probability cut if needed
                    probabilities_list = []
                    if proba_cut:
                        keep_index = []
                        for j in range(len(pred_prob_query)):
                            if (pred_prob_query[j][0] > 0.4) & (
                                pred_prob_query[j][0] < 0.6
                            ):
                                keep_index.append(j)
                                probabilities_list.append(float(pred_prob_query[j][0]))
                        new_alerts_use = alerts_use.iloc[keep_index]
                        if verbose:
                            print(probabilities_list)
                    else:
                        new_alerts_use = alerts_use
                        probabilities_list = pred_prob_query[:, 0].flatten()

                    if len(new_alerts_use) > 0:

                        alerts_format_pd[
                            alerts_format_pd["objectId"].isin(
                                new_alerts_use["objectId"].unique()
                            )
                        ].to_csv(
                            dir_output
                            + date
                            + "/"
                            + strategy
                            + "/queries/alert_data.csv",
                            index=False,
                        )

                        if verbose:
                            print(f"New alerts to use {len(new_alerts_use)}")

                        if plot_lcs:
                            for i, lcname in enumerate(
                                new_alerts_use["objectId"].unique()
                            ):
                                lc = alerts_format_pd[
                                    alerts_format_pd["objectId"] == lcname
                                ]
                                proba = probabilities_list[i]
                                pu.plot_lc_mag(lc.reset_index(), proba, dir_suffix)
                                pu.plot_lc_flux_wfit(
                                    alerts_flux[alerts_flux["id"] == lcname],
                                    proba,
                                    alerts_features,
                                    dir_suffix,
                                )

                        # update training and save
                        train_updated = pd.concat(
                            [train_for_loop, new_alerts_use], ignore_index=True
                        )
                        train_updated.to_csv(
                            dir_output
                            + date
                            + "/"
                            + strategy
                            + "/training_samples/train.csv",
                            index=False,
                        )
                        if verbose:
                            print("    train_updated.shape = ", train_updated.shape)

                        # save probabilities
                        proba_out = np.append(
                            train_for_loop_probas["probability"].values,
                            probabilities_list,
                        )
                        train_for_loop_probas = pd.DataFrame()
                        train_for_loop_probas["objectId"] = train_updated[
                            "objectId"
                        ].values
                        train_for_loop_probas["probability"] = proba_out
                        train_for_loop_probas.to_csv(
                            dir_output
                            + date
                            + "/"
                            + strategy
                            + "/training_samples/probabilities.csv",
                            index=False,
                        )

                        # train model and save
                        clf = RandomForestClassifier(random_state=42, n_estimators=1000)
                        clf.fit(
                            train_updated[list(train_updated.keys())[:-3]],
                            train_updated["type"].values == "Ia",
                        )
                        pickle.dump(
                            clf,
                            open(
                                dir_output
                                + date
                                + "/"
                                + strategy
                                + "/models/model.pkl",
                                "wb",
                            ),
                        )

                        # make predictions and save
                        pred_prob = clf.predict_proba(
                            test_for_loop[list(test_for_loop.keys())[:-2]]
                        )
                        pred_prob_pd = pd.DataFrame(
                            np.hstack(
                                [
                                    test_for_loop["objectId"].values.reshape(-1, 1),
                                    np.array(pred_prob[:, 1]).reshape(-1, 1),
                                ]
                            ),
                            columns=["objectId", "probIa"],
                        )
                        pred_prob_pd.to_csv(
                            dir_output
                            + date
                            + "/"
                            + strategy
                            + "/class_prob/test_class_prob.csv",
                            index=False,
                        )

                        # calculate metrics
                        pred = clf.predict(
                            test_for_loop[list(test_for_loop.keys())[:-2]]
                        )
                        names, res = get_snpcc_metric(
                            pred, test_for_loop["type"].values == "Ia"
                        )
                        if verbose:
                            print("   res = ", res)

                        metric_pd = pd.DataFrame(
                            [
                                [date]
                                + [list(new_labels.keys()).index(date) + 30]
                                + res
                                + [list(alerts_use["objectId"].values)]
                            ],
                            columns=["date", "loop"] + names + ["query_objectIds"],
                        )
                        metric_pd["date_plot"] = d
                        metric_pd.to_csv(
                            dir_output + date + "/" + strategy + "/metrics/metric.csv",
                            index=False,
                        )
                        metrics_all_list.append(metric_pd)

    if len(metrics_all_list) > 0:
        # if this date had a retraining, append metrics
        metrics = pd.concat(metrics_all_list, ignore_index=True)
        metrics["n spectra"] = (
            metrics["query_objectIds"].apply(lambda x: len(x)).cumsum()
        )

    else:
        metrics = pd.DataFrame()
    metrics.to_csv(dir_output + "metrics.csv", index=False)

    return metrics


def convert_DF_dic_labels(df, key_spectra="discoveryjd+9"):
    """TNS labels DataFrame to dictionary

    Args:
        df (pd.DataFrame): TNS classifications (Fink format)

    Returns:
        dictionary : formated labels in dictionary for AL loop
    """
    # convert to new_labels format
    dic_labels = {}
    for dat in df[key_spectra].unique()[:60]:
        sel = df[df[key_spectra] == dat]

        # add one day for tag acquisition
        sel.loc[:, key_spectra] += 1
        tmp_arr = sel[["ztf_names", "type AL", key_spectra]].to_numpy()

        date_in_str_fmt = sel[f"{key_spectra}_strfmt"].values[0]
        dic_labels[date_in_str_fmt] = tmp_arr

    return dic_labels


def convert_DF_dic_labels_old(df):
    """TNS labels DataFrame to dictionary

    Args:
        df (pd.DataFrame): TNS classifications (Fink format)

    Returns:
        dictionary : formated labels in dictionary for AL loop
    """
    # convert to new_labels format
    dic_labels = {}
    for dat in df["discoveryjd+9"].unique()[:60]:
        sel = df[df["discoveryjd+9"] == dat]
        tmp_arr = sel[["ztf_names", "type AL", "discoveryjd+9"]].to_numpy()

        date_in_str_fmt = sel["discoveryjd+9_strfmt"].values[0]
        dic_labels[date_in_str_fmt] = tmp_arr

    return dic_labels
