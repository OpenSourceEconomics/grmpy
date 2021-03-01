"""This module contains methods for producing the estimation output files."""
import time
from textwrap import wrap


def print_logfile(init_dict, rslt, print_output):
    """The function writes the log file for the estimation process."""
    # Adjust output
    if "output_file" in init_dict["ESTIMATION"].keys():
        file_name = init_dict["ESTIMATION"]["output_file"]
    else:
        file_name = "est.grmpy.info"
    file_input = ""

    # obtain optimization information and results
    opt_info = rslt["opt_info"]
    opt_rslt = rslt["opt_rslt"]

    # create header of output table
    header = "{:>50}".format("Optimization Results") + "\n" + 80 * "=" + "\n"

    # specify some table inputs
    message = wrap(opt_info["message"], 24)
    if len(message) == 1:
        message += [""]
    time_now = time.localtime()
    time_ = time.strftime("%H:%M:%S", time_now)
    date = time.strftime("%a, %d %b %Y", time_now)

    # define correctly aligned string
    fmt = "{:<15}" + "{:>20}" + "{:>5}" + "{:<16}" + "{:>24}\n"

    # define table input
    header_input = [
        ("Dep. Variable:", opt_info["dependent"], "Optimizer:", opt_info["optimizer"]),
        ("Choice Var:", opt_info["indicator"], "No. Evaluations:", opt_info["nit"]),
        ("Date:", date, "Success:", opt_info["success"]),
        ("Time:", time_, "Status:", opt_info["status"]),
        ("Observations:", opt_info["observations"], "Message:", message[0]),
        ("Start Values:", opt_info["start"], "", message[1]),
    ]

    # loop over header input

    for line in header_input:
        header += fmt.format(line[0], line[1], "", line[2], line[3])

    header += fmt.format("", "", "", "Criterion Func:", "")
    fmt = "{:<15}" + "{:>20}" + "{:>10}" + "{:<16}" + "{:>+19.4f}\n"
    for section in [("Start", opt_info["crit"]), ("Finish", opt_info["crit"])]:
        header += fmt.format("", "", "", section[0] + ":", section[1])

    header += "=" * 80 + "\n"

    # Add estimation output
    fmt = "{:<17}" + "{:>11}" + "{:>10}" * 3 + "{:>11}" + "{:>11}" "\n"

    header += fmt.format("", "coef", "std err", "t", "P>|t|", "[0.025", "0.975]")
    header += "-" * 80

    estimation_output = write_identifier_section(opt_rslt, opt_info)

    file_input = header + estimation_output

    if print_output is True:
        print(file_input)
    with open(file_name, "w") as file_:
        file_.write(file_input)


def write_identifier_section(opt_rslt, opt_info):
    """This function prints the information about the estimation results in the output
    file.
    """

    # write estimation output
    est_out = ""
    fmt_section = "\n{:<14}\n\n"
    fmt = (
        "{:<17.17}"
        + "{:>11.4f}"
        + "{:>10.3f}"
        + "{:>10.3f}"
        + "{:>10.3f}"
        + "{:>11.3f}" * 2
        + "\n"
    )
    for section in ["TREATED", "UNTREATED", "CHOICE", "DIST"]:
        est_out += fmt_section.format(section)
        params = opt_rslt.loc[section, "params"]
        std = opt_rslt.loc[section, "std"]
        t = opt_rslt.loc[section, "t_values"]
        P = opt_rslt.loc[section, "p_values"]
        conf_int_low = opt_rslt.loc[section, "conf_int_low"]
        conf_int_up = opt_rslt.loc[section, "conf_int_up"]

        for var_label in opt_rslt.loc[section, slice(None)].index:
            est_out += fmt.format(
                var_label,
                params.loc[var_label],
                std.loc[var_label],
                t.loc[var_label],
                P.loc[var_label],
                conf_int_low.loc[var_label],
                conf_int_up.loc[var_label],
            )
    est_out += 80 * "=" + "\n"

    # Add warnings to table

    est_out += "\n{}\n".format("Warning:")

    for i in opt_info["warning"]:
        est_out += "\n"
        lines = wrap(i, 80)
        for j in lines:
            est_out += f"{j}\n"

    return est_out
