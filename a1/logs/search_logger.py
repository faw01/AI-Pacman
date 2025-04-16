"""
Logger for single search agent.

Logs timestamps, functions called, state of agent, locayion of agent, path length, expansion, #capsules eaten,
#fruits eaten, #ghosts run into, #wins, #losses, avg. score, scores, win rate

@author: emelypi
@date: 22.07.2023
"""

import functools
import logging as log



def _flush():
    for function_name, log_string in log_function.buffer:
          log.getLogger(function_name).info(log_string)
    log_function.buffer.clear()

def log_function(func: callable):
    """
    a log decorator for writing the inputs parameters and the return value of a function to a file.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
            if log_function.remaining_log_calls <= 0: return func(*args, **kwargs)

            log_function.buffer.append((func.__name__, f"I: {','.join(list(map(str, args)))},{','.join(list(map(str, kwargs)))}"))
            result = func(*args, **kwargs)
            log_function.buffer.append((func.__name__, f"O: {result}"))

            log_function.remaining_log_calls -= 1
            return result
    
    log_function.buffer = []
    log_function.remaining_log_calls = 1000
    log_function.flush = _flush
    return wrapper


def search_logger(filename: str = None):

    logger = log.getLogger('root')

    if filename is None:
        logger.disabled = True
        return

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if filename == "":
        c_handler = log.StreamHandler()
        c_handler.setLevel(log.INFO)
        c_format = log.Formatter(log_format)
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

    else:
        f_handler = log.FileHandler('logs/'+filename+'.log')
        f_handler.setLevel(log.INFO)
        f_format = log.Formatter(log_format)
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    logger.setLevel(log.INFO)
    logger.info("Logger Version: 2.0.0")



