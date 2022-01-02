
import os
import sys
import torch
import logging
import traceback


def get_output_dim(model, pooling_type="gem"):
    """Dinamically compute the output size of a model.
    """
    output_dim = model(torch.ones([2, 3, 224, 224])).shape[1]
    if pooling_type == "netvlad":
        output_dim *= 64  # NetVLAD layer has 64x bigger output dimensions
    return output_dim


class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()
    def __iter__(self):
        return self
    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


def setup_logging(output_folder, exist_ok=False, console="debug",
                  info_filename="info.log", debug_filename="debug.log"):
    """Set up logging files and console output.
    Creates one file for INFO logs and one for DEBUG logs.
    Args:
        output_folder (str): creates the folder where to save the files.
        exist_ok (boolean): if False throw a FileExistsError if output_folder already exists
        debug (str):
            if == "debug" prints on console debug messages and higher
            if == "info"  prints on console info messages and higher
            if == None does not use console (useful when a logger has already been set)
        info_filename (str): the name of the info file. if None, don't create info file
        debug_filename (str): the name of the debug file. if None, don't create debug file
    """
    if not exist_ok and os.path.exists(output_folder):
        raise FileExistsError(f"{output_folder} already exists!")
    os.makedirs(output_folder, exist_ok=True)
    # logging.Logger.manager.loggerDict.keys() to check which loggers are in use
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('shapely').disabled = True
    logging.getLogger('shapely.geometry').disabled = True
    base_formatter = logging.Formatter('%(asctime)s   %(message)s', "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    logging.getLogger('PIL').setLevel(logging.INFO)  # turn off logging tag for some images
    
    if info_filename != None:
        info_file_handler = logging.FileHandler(f'{output_folder}/{info_filename}')
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(base_formatter)
        logger.addHandler(info_file_handler)
    
    if debug_filename != None:
        debug_file_handler = logging.FileHandler(f'{output_folder}/{debug_filename}')
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(base_formatter)
        logger.addHandler(debug_file_handler)
    
    if console != None:
        console_handler = logging.StreamHandler()
        if console == "debug": console_handler.setLevel(logging.DEBUG)
        if console == "info":  console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(base_formatter)
        logger.addHandler(console_handler)
    
    def exception_handler(type_, value, tb):
        logger.info("\n" + "".join(traceback.format_exception(type, value, tb)))
    sys.excepthook = exception_handler

