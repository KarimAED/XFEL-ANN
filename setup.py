import sys
import os

import numpy as np
import pandas as pd

from delay_prep import save_delays
from emean_prep import save_emean


def get_inputs(source, files):
    # init empty placeholders
    ebeam_labels = []
    ebeam_data = []
    epic_labels = []
    epic_data = []
    gmd_labels = []
    gmd_data = []

    for f in files:
        if any([i in f for i in ["EBeam", "EPICS", "GMD"]]):  # avoid unnecessary data loading
            data = np.load(os.path.join(source, f))
        if "EBeam" in f:
            if not ebeam_labels:
                ebeam_data = data["EBeamValuesList"]
                ebeam_labels = data["EBeamParameterNames"].tolist()
            else:
                ebeam_data = np.append(ebeam_data, data["EBeamValuesList"], axis=0)
        elif "EPICS" in f:
            if not epic_labels:
                epic_data = data["EPICSValuesList"]
                epic_labels = data["EPICSParameterNames"].tolist()
            else:
                epic_data = np.append(epic_data, data["EPICSValuesList"], axis=0)
        elif "GMD" in f:
            if not gmd_labels:
                gmd_data = data["GMDValuesList"]
                gmd_labels = data["GMDParameterNames"].tolist()
            else:
                gmd_data = np.append(gmd_data, data["GMDValuesList"], axis=0)

    labels = ebeam_labels + epic_labels + gmd_labels
    labels = [str(label, "utf-8") for label in labels]
    temp_data = np.append(ebeam_data, epic_data, axis=1)
    data = np.append(temp_data, gmd_data, axis=1)

    return data, labels


def format_single(source, target):

    files = os.listdir(source)

    inp, inp_labels = get_inputs(source, files)

    # initialise output features
    optical_data = []
    optical_labels = []

    for f in files:
        if "Optical" in f:
            data = np.load(os.path.join(source, f), allow_pickle=True, encoding="latin1")
            sf_lam = lambda x: np.array([0, 0, 0]) if x is None else x
            sf_list = [sf_lam(i) for i in data["UXSSingleFitList"]]
            sf_data = np.stack(sf_list)
            if not optical_labels:
                optical_labels = ["GaussAmp", "GaussMean_pxl", "GaussStd_pxl", "FitMask"]\
                                 + data["xUXS"].astype("str").tolist()

                fit_info = np.append(sf_data, np.array([data["UXSSingleFitListMask"]]).T, axis=1)
                optical_data = np.append(fit_info, data["UXSProfileList"], axis=1)
            else:
                fit_info = np.append(sf_data, np.array([data["UXSSingleFitListMask"]]).T, axis=1)
                data = np.append(fit_info, data["UXSProfileList"], axis=1)
                optical_data = np.append(optical_data, data, axis=0)

    d = os.getcwd()
    if not os.path.exists(target):
        os.mkdir(target)
    os.chdir(target)
    np.savetxt("single_inputs.tsv.gz", inp, delimiter="\t", header="\t".join(inp_labels), comments="")
    np.savetxt("single_outputs.tsv.gz", optical_data, delimiter="\t", header="\t".join(optical_labels), comments="")
    os.chdir(d)


def format_double(source, target):
    files = os.listdir(source)

    inp, inp_labels = get_inputs(source, files)

    # initialise output features
    delay_data = []
    delay_labels = []
    tof_data = []
    tof_labels = []

    for f in files:
        if "Delay" in f:
            data = np.load(os.path.join(source, f))
            if not delay_labels:
                delay_labels = ["Delays", "DelayMask"]
                delay_data = np.array([data["DelayValuesList"].flatten(), data["DelayValuesListMask"]]).T
            else:
                temp_data = np.array([data["DelayValuesList"].flatten(), data["DelayValuesListMask"]]).T
                delay_data = np.append(delay_data, temp_data, axis=0)

        elif "TOF" in f:
            data = np.load(os.path.join(source, f))
            if not tof_labels:
                tof_labels = ["LowGaussAmp", "LowGaussMean_eV", "LowGaussStd_eV", "HighGaussAmp", "HighGaussMean_eV",
                              "HighGaussStd_eV", "DelaysMask"] + data["xTOF"].astype("str").tolist()

                fit_info = np.append(data["TOFDoubleFitList"], np.array([data["TOFDoubleFitListMask"]]).T, axis=1)
                tof_data = np.append(fit_info, data["TOFProfileList"], axis=1)

            else:
                fit_info = np.append(data["TOFDoubleFitList"], np.array([data["TOFDoubleFitListMask"]]).T, axis=1)
                temp_data = np.append(fit_info, data["TOFProfileList"], axis=1)
                tof_data = np.append(tof_data, temp_data, axis=0)

    output = np.append(delay_data, tof_data, axis=1)
    output_labels = delay_labels + tof_labels

    d = os.getcwd()
    if not os.path.exists(target):
        os.mkdir(target)
    os.chdir(target)
    np.savetxt("double_inputs.tsv.gz", inp, delimiter="\t", header="\t".join(inp_labels), comments="")
    np.savetxt("double_outputs.tsv.gz", output, delimiter="\t", header="\t".join(output_labels), comments="")
    os.chdir(d)

def format_new(source, target):
    fname = os.path.join(source, "Run203_fixed.pkl.gz")
    new_df = pd.read_pickle(fname)

    inp_df = pd.DataFrame()
    out_df = pd.DataFrame()

    for name, column in new_df.iteritems():
        inp = [name.startswith(i) and (not "DESC" in name) for i in ["epic_", "ebeam", "f_"]]
        if any(inp):
            inp_df[name] = column

        if name == "XTCAV_pump_probe_delay":
            out_df["Delays"] = column
            out_df["DelayMask"] = out_df["Delays"] > -20

    d = os.getcwd()
    if not os.path.exists(target):
        os.mkdir(target)
    os.chdir(target)
    np.savetxt("new_inputs.tsv.gz", inp_df.values.astype(np.float64),
               delimiter="\t", header="\t".join(inp_df.columns), comments="")
    np.savetxt("new_outputs.tsv.gz", out_df.values.astype(np.float64),
               delimiter="\t", header="\t".join(out_df.columns), comments="")

if __name__ == "__main__":
    f_path = os.path.dirname(sys.argv[0])
    target_dir = os.path.join(f_path, "Data")
    if not os.path.exists(os.path.join(f_path, "DataLCLS2017")):
        os.system("git clone https://github.com/alvarosg/DataLCLS2017.git")  # original data

    data_dir = os.path.join(f_path, "DataLCLS2017/Data/")
    single_dir = os.path.join(data_dir, "amof6215")
    double_dir = os.path.join(data_dir, "amo86815")
    new_dir = os.path.join(f_path, "DataLCLSNew")
    if os.path.normpath(f_path) != ".":
        os.chdir(f_path)
    #format_single(single_dir, target_dir)
    #format_double(double_dir, target_dir)
    print("Loading DataLCLSNew...")
    format_new(new_dir, target_dir)
    print("Done.")
    #save_delays(f_path, 0.15)
    print("Preprocessing DataLCLSNew...")
    save_delays(f_path, 0.15, new=True)
    print("Done.")
    #save_emean(f_path, 0.15)
