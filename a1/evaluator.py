import glob
import os
import subprocess
import sys
from itertools import chain, product
from optparse import OptionParser
from typing import Dict, List



import json
import os
import re
import subprocess
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

    parser.add_option('--q1a', help='Whether to run q1a or not', dest='q1a', action='store_false', default=True)
    parser.add_option('--q1b', help='Whether to run q1b or not', dest='q1b', action='store_false', default=True)
    parser.add_option('--q1c', help='Whether to run q1c or not', dest='q1c', action='store_false', default=True)
    parser.add_option('--q2', help='Whether to run q2 or not', dest='q2', action='store_false', default=True)

    options, otherjunk = parser.parse_args(argv)
    args = dict()

    args['q1a'] = options.q1a
    args['q1b'] = options.q1b
    args['q1c'] = options.q1c
    args['q2'] = options.q2

    return args


if __name__ == "__main__":

    args = readCommand(sys.argv[1:])

    logs_dir = './logs/'
    logs = glob.glob(logs_dir + "*.log")
    for log in logs: os.remove(log)

    layouts_dir = "./layouts/"
    question_1a_pattern = "q1a_*.lay"
    question_1a_layouts = glob.glob(layouts_dir + question_1a_pattern)

    
    question_1b_pattern = "q1b_*.lay"
    question_1b_layouts = glob.glob(layouts_dir + question_1b_pattern)

    
    question_1c_pattern = "q1c_*.lay"
    question_1c_layouts = glob.glob(layouts_dir + question_1c_pattern)

    
    question_2_patterns = "q2_*.lay"
    question_2_layouts = glob.glob(layouts_dir + question_2_patterns)

    question_1a_setup: Dict = {
        'layout': question_1a_layouts,
        'average_score': None,
        'win_rate': None,
    }

    question_1b_setup: Dict = {
        'layout': question_1b_layouts,
        'average_score': None,
        'win_rate': None,
    }

    question_1c_setup: Dict = {
        'layout': question_1c_layouts,
        'average_score': None,
        'win_rate': None,
    }

    question_2_setup: Dict = {
        'layout': question_2_layouts,
        'average_score': None,
        'win_rate': None,
    }

    if disclaimer() != "y":
        print("")
        exit()

    question_1a = pd.DataFrame(question_1a_setup)
    question_1b = pd.DataFrame(question_1b_setup)
    question_1c = pd.DataFrame(question_1c_setup)
    question_2 = pd.DataFrame(question_2_setup)

    # Question 1a
    if args["q1a"]:
        for index, row in (t := tqdm(question_1a.iterrows(), total=question_1a.shape[0])):
            if not os.path.isfile(row['layout']): continue

            t.set_description(f"Running Q1a:{row['layout']}")
            command = ['python', 'pacman.py', '-l', row['layout'], '-p', 'SearchAgent', '-a',
                       'fn=q1a_solver,prob=q1a_problem', '--timeout=1', '-q',
                       '-o', os.path.splitext(os.path.basename(row['layout']))[0]]
            result = run(command)

            re_match = re.search(r"Average\sScore:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
            question_1a.at[index, 'average_score'] = re_match.group(1) if re_match else None

            re_match = re.search(r"Win\sRate:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
            question_1a.at[index, 'win_rate'] = re_match.group(1) if re_match else None

    # Question 1b
    if args["q1b"]:
        for index, row in (t := tqdm(question_1b.iterrows(), total=question_1b.shape[0])):
            if not os.path.isfile(row['layout']): continue

            t.set_description(f"Running Q1b:{row['layout']}")
            command = ['python', 'pacman.py', '-l', row['layout'], '-p', 'SearchAgent', '-a',
                       'fn=q1b_solver,prob=q1b_problem', '--timeout=5', '-q',
                       '-o', os.path.splitext(os.path.basename(row['layout']))[0]]
            result = run(command)

            re_match = re.search(r"Average\sScore:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
            question_1b.at[index, 'average_score'] = re_match.group(1) if re_match else None

            re_match = re.search(r"Win\sRate:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
            question_1b.at[index, 'win_rate'] = re_match.group(1) if re_match else None

    # Question 1c
    if args["q1c"]:
        for index, row in (t := tqdm(question_1c.iterrows(), total=question_1c.shape[0])):
            if not os.path.isfile(row['layout']): continue

            t.set_description(f"Running Q1c:{row['layout']}")
            command = ['python', 'pacman.py', '-l', row['layout'], '-p', 'SearchAgent', '-a',
                       'fn=q1c_solver,prob=q1c_problem', '--timeout=10', '-q',
                       '-o', os.path.splitext(os.path.basename(row['layout']))[0]]
            result = run(command)

            re_match = re.search(r"Average\sScore:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
            question_1c.at[index, 'average_score'] = re_match.group(1) if re_match else None

            re_match = re.search(r"Win\sRate:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
            question_1c.at[index, 'win_rate'] = re_match.group(1) if re_match else None

    # Question 2
    if args["q2"]:
        for index, row in (t := tqdm(question_2.iterrows(), total=question_2.shape[0])):
            if not os.path.isfile(row['layout']): continue

            t.set_description(f"Running Q2:{row['layout']}")
            command = ['python', 'pacman.py', '-l', row['layout'], '-p', 'Q2_Agent', '--timeout=30', '-q', '-f',
                       '-o', os.path.splitext(os.path.basename(row['layout']))[0]]
            result = run(command)

            re_match = re.search(r"Average\sScore:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
            question_2.at[index, 'average_score'] = re_match.group(1) if re_match else None

            re_match = re.search(r"Win\sRate:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
            question_2.at[index, 'win_rate'] = re_match.group(1) if re_match else None


    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)

    print("\nEvaluation Report")
    print("=" * 160)
    print(f"Question 1a Results:\n{question_1a.to_markdown()}\n")
    print(f"Question 1b Results:\n{question_1b.to_markdown()}\n")
    print(f"Question 1c Results:\n{question_1c.to_markdown()}\n")
    print(f"Question 2 Results:\n{question_2.to_markdown()}\n")

    print("=" * 160)

    
