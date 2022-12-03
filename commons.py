
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


def setup_logging(output_folder, console="info", info_filename="info.log",
                  debug_filename="debug.log"):
    """After calling this function, you can easily log messages from anywhere
    in your code without passing any object to your functions.
    Just calling "logging.info(msg)" prints "msg" in stdout and saves it in
    the "info.log" and "debug.log" files.
    Similarly, "logging.debug(msg)" saves "msg" in the "debug.log" file.
    Exceptions raised from any other function are also saved in the files
    info.log and debug.log.
    
    Parameters
    ----------
    output_folder : str, the folder where to save the logging files.
    console : str, can be "debug" or "info".
        If console == "debug", print in stdout any time logging.debug(msg)
        (or logging.info(msg)) is called.
        If console == "info", print in std out only when logging.info(msg) is called.
    info_filename : str, name of the file with the logs printed when calling
        logging.info(msg).
    debug_filename : str, name of the file with the logs printed when calling
        logging.debug(msg) or logging.info(msg).
    
    """
    if os.path.exists(output_folder):
        raise FileExistsError(f"{output_folder} already exists!")
    os.makedirs(output_folder)
    
    # Use logging.Logger.manager.loggerDict.keys() to check which loggers are in use
    # Disable some useless warnings.
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('shapely').disabled = True
    logging.getLogger('shapely.geometry').disabled = True
    base_formatter = logging.Formatter('%(asctime)s   %(message)s', "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    logging.getLogger('PIL').setLevel(logging.INFO)  # turn off logging tag for some images
    
    if info_filename is not None:
        info_file_handler = logging.FileHandler(f'{output_folder}/{info_filename}')
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(base_formatter)
        logger.addHandler(info_file_handler)
    
    if debug_filename is not None:
        debug_file_handler = logging.FileHandler(f'{output_folder}/{debug_filename}')
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(base_formatter)
        logger.addHandler(debug_file_handler)
    
    if console is not None:
        console_handler = logging.StreamHandler()
        if console == "debug":
            console_handler.setLevel(logging.DEBUG)
        if console == "info":
            console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(base_formatter)
        logger.addHandler(console_handler)
    
    # Save exceptions in log files
    def exception_handler(type_, value, tb):
        logger.info("\n" + "".join(traceback.format_exception(type, value, tb)))
    
    sys.excepthook = exception_handler
