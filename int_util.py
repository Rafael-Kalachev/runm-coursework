import shutil
import os
import pandas as pd 

from pandas import DataFrame

import config as _conf


def prepare_output_dir(output_dir):
    # type: (str) -> None
    """! Prepares the output direcotry: 
            - Creates the the output directoy if not created.
            - Removes the content of the direcotry it not empty.
        @param output_dir   the directory that will be prepared

        @return None
    """
    
    shutil.rmtree(output_dir,ignore_errors=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def get_dataset_size(df):
    # type: (DataFrame) -> list(int)
    """! Get the size of the dataframe 
    
        @param df   dataframe

        @result     size of the dataframe
    """

    return df.shape

def load_dataframe_from_file(file_path):
    # type: (str) -> DataFrame
    """! Load the dataframe form a given file 
    
        @param file_path    path to the XLS/CSV file to be loaded
        
        @result     Dataframe containg the data of the file passed
    """

    return pd.read_excel(file_path)

def log(msg, prepend="", file_output=_conf.run_log_file_path, show_stdout=True):
    # type: (str, str, str, bool) -> None
    """! Log message to stdout and file

        @param msg          string containing the message to be printed
        @param prepend      string that will be printed before the message
        @param file_output  path to a file in which the output will be appended (if None passed this will be skipped)
        @param show_sdtout  directs if the message shal be printed to stdout or not

        @return None
    """

    final_output = "{}{}".format(prepend, msg)
    if show_stdout:
        print(final_output)

    if file_output is not None:
        with open(file_output, "a") as file:
            file.write(final_output)
            file.write("\n")
    

