import glob
import os
import subprocess
import sys
from itertools import chain, product
from optparse import OptionParser

import json
import re
from typing import Dict, List

import pandas as pd
from tqdm import tqdm


def disclaimer() -> bool:
    message = """
    -------------------------------------------------------------------------------
                                    ATTENTION

    Please ensure you are up to date with the latest code changes. Failing to stay 
    updated with the latest code changes puts your work at risk of not being 
    evaluated correctly.
    -------------------------------------------------------------------------------
    I CONFIRM I HAVE PULLED THE LATEST VERSION OF ASSIGNMENT: [y/N] """

    return input(message)

def linear_product(parameters: Dict) -> List[str]:
    for experiment in product(*parameters.values()):
        yield list(chain(*zip(parameters, experiment)))


def run(command: List[str]) -> subprocess.CompletedProcess:
    """
    Runs a command and returns the completed process.

    Args:
        command (List[str]): The command to run.

    Returns:
        subprocess.CompletedProcess: The completed process.
   """
    try:
        retval = subprocess.run(command, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        print(e.stderr.decode('utf-8'))
        exit(1)

    return retval

def readCommand( argv ):
    """
    Processes the command used to run pacman from the command line.
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:      python evalutor.py <options>
    """
    parser = OptionParser(usageStr)

    parser.add_option('--q1', help='Whether to run q1 or not', dest='q1', action='store_false', default=True)
    parser.add_option('--q2', help='Whether to run q2 or not', dest='q2', action='store_false', default=True)
    parser.add_option('--q3', help='Whether to run q3 or not', dest='q3', action='store_false', default=True)

    options, otherjunk = parser.parse_args(argv)
    args = dict()

    args['q1'] = options.q1
    args['q2'] = options.q2
    args['q3'] = options.q3

    return args


def get_q1_parameters(layout_name, parameters_json):
    if "VI_small" in layout_name:
        return parameters_json["small"]["gamma"], 50
    elif "VI_medium" in layout_name:
        return parameters_json["medium"]["gamma"], 100
    else:
        return parameters_json["big"]["gamma"], 150

def get_q2_parameters(layout_name, parameters_json):
    if "QL_tiny" in layout_name:
        return parameters_json["tiny"]["epsilon"], parameters_json["tiny"]["alpha"], parameters_json["tiny"]["gamma"], 100
    elif "QL_small" in layout_name:
        return parameters_json["small"]["epsilon"], parameters_json["small"]["alpha"], parameters_json["small"]["gamma"], 200
    else:
        return parameters_json["medium"]["epsilon"], parameters_json["medium"]["alpha"], parameters_json["medium"]["gamma"], 300
    

if __name__ == "__main__":

    args = readCommand(sys.argv[1:])

    logs_dir = './logs/'
    logs = glob.glob(logs_dir + "*.log")
    for log in logs: os.remove(log)

    with open("./models/Q1_parameters.json") as params:
        q1_parameters_json = json.load(params)

    with open("./models/Q2_parameters.json") as params:
        q2_parameters_json = json.load(params)

    layouts_dir = "./layouts/"
    question_1_pattern = "VI_*.lay"
    question_1_layouts = glob.glob(layouts_dir + question_1_pattern)

    question_2_pattern = "QL_*.lay"
    question_2_layouts = glob.glob(layouts_dir + question_2_pattern)

    question_3_pattern = "ML_*.lay"
    question_3_layouts = glob.glob(layouts_dir + question_3_pattern)

    question_1_setup: Dict = {
        'layout': question_1_layouts,
        'average_score': [None] * len(question_1_layouts),
        'win_rate': [None] * len(question_1_layouts),
    }

    question_2_setup: Dict = {
        'layout': question_2_layouts,
        'average_score': [None] * len(question_2_layouts),
        'win_rate': [None] * len(question_2_layouts),
    }

    question_3_setup: Dict = {
        'layout': question_3_layouts,
        'average_score': [None] * len(question_3_layouts),
        'win_rate': [None] * len(question_3_layouts),
    }

    if disclaimer() != "y":
        print("")
        exit()

    question_1 = pd.DataFrame(question_1_setup)
    question_2 = pd.DataFrame(question_2_setup)
    question_3 = pd.DataFrame(question_3_setup)
    
    question_1["gamma"], question_1["iterations"] = zip(*question_1["layout"].apply(lambda l: get_q1_parameters(l, q1_parameters_json)))
    question_2["epsilon"], question_2["alpha"], question_2["gamma"], question_2["numTrainingEpisodes"] = zip(*question_2["layout"].apply(lambda l: get_q2_parameters(l, q2_parameters_json)))
    
    question_1 = question_1.sort_values(by="layout").reset_index(drop=True)
    question_2 = question_2.sort_values(by="layout").reset_index(drop=True)
    question_3 = question_3.sort_values(by="layout").reset_index(drop=True)

    # Question 1a
    if args["q1"]:
        for index, row in (t := tqdm(question_1.iterrows(), total=question_1.shape[0])):
            if not os.path.isfile(row['layout']): continue

            t.set_description(f"Running Q1:{row['layout']}")
            command = ['python', 'pacman.py', '-l', row['layout'], '-p', 'Q1Agent', '-a',
            f'discount={row["gamma"]},iterations={row["iterations"]}', '--timeout=1', '-q', '-g', 'StationaryGhost', '-n', '40', '-f']
            result = run(command)

            re_match = re.search(r"Average\sScore:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
            question_1.at[index, 'average_score'] = re_match.group(1) if re_match else None

            re_match = re.search(r"Win\sRate:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
            question_1.at[index, 'win_rate'] = re_match.group(1) if re_match else None

    # Question 2
    if args["q2"]:
        for index, row in (t := tqdm(question_2.iterrows(), total=question_2.shape[0])):
            if not os.path.isfile(row['layout']): continue

            t.set_description(f"Running Q2:{row['layout']}")
            command = ['python', 'pacman.py', '-l', row['layout'], '-p', 'Q2Agent', '-a',
                    f'gamma={row["gamma"]},epsilon={row["epsilon"]},alpha={row["alpha"]}',
                      '--timeout=5', '-q', '-g', 'StationaryGhost', '-x', f'{row["numTrainingEpisodes"]}', '-n', f'{row["numTrainingEpisodes"] + 1}', '-f']
            result = run(command)

            re_match = re.search(r"Average\sScore:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
            question_2.at[index, 'average_score'] = re_match.group(1) if re_match else None

            re_match = re.search(r"Win\sRate:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
            question_2.at[index, 'win_rate'] = re_match.group(1) if re_match else None

    # Question 1c
    if args["q3"]:
        for index, row in (t := tqdm(question_3.iterrows(), total=question_3.shape[0])):
            if not os.path.isfile(row['layout']): continue

            t.set_description(f"Running Q1c:{row['layout']}")
            command = ['python', 'pacman.py', '-l', row['layout'], '-p', 'Q3Agent', '--timeout=5', '-q', '-n', '40', '-f']
            result = run(command)

            re_match = re.search(r"Average\sScore:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
            question_3.at[index, 'average_score'] = re_match.group(1) if re_match else None

            re_match = re.search(r"Win\sRate:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
            question_3.at[index, 'win_rate'] = re_match.group(1) if re_match else None


    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)

    print("\nEvaluation Report")
    print("=" * 160)
    if args["q1"]: print(f"Question 1 Results:\n{question_1.to_markdown()}\n")
    if args["q2"]: print(f"Question 2 Results:\n{question_2.to_markdown()}\n")
    if args["q3"]: print(f"Question 3 Results:\n{question_3.to_markdown()}\n")

    print("=" * 160)

    