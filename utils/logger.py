import os
from datetime import datetime
import pytz
from utils import file_utils
import logging
import sys
import re
from rdkit import RDLogger
def get_curr_time():
    return datetime.now(pytz.timezone('Europe/London')).strftime('%b%d(%H-%M-%S)')


class Logger:
    def __init__(self, config, is_train=True, show_exc=False, level=logging.INFO):

        self.config = config
        self.data_name = config.data.name
        self.exp_name = config.exp_name
        self.exp_time = get_curr_time()

        pl = file_utils.PathLoader(config, is_train, self.exp_time)  # it will create essential folders
        # Only log paths live here; everything else is owned by PathLoader.
        self.log_dir = pl.log_dir
        self.log_name = pl.ckpt_name if is_train else f'{pl.ckpt_name}(sample at {self.exp_time})'
        if os.path.isabs(self.log_dir):
            print(f"log_dir is absolute path: {self.log_dir}")
            self.log_name = re.sub(r'_\d+$', '', os.path.basename(self.log_name))
        self.log_path = os.path.join(self.log_dir, self.log_name + '.log')

        self.logger = logging.getLogger()
        self.logger.setLevel(level)
        self.formatter = logging.Formatter('%(asctime)s - %(filename)s(L%(lineno)d) - %(message)s',
                                           datefmt='%Y-%m-%d %H:%M')

        self.file_handler = logging.FileHandler(self.log_path)
        self.file_handler.setLevel(level)
        self.file_handler.setFormatter(self.formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)

        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(console_handler)

        RDLogger.DisableLog('rdApp.error')
        RDLogger.DisableLog('rdApp.warning')

        MY_DEVICE = os.getenv('MY_DEVICE', None)
        if MY_DEVICE is not None:
            self.log(f'Using MY_DEVICE={MY_DEVICE}')

        if show_exc:
            def handle_exception(exc_type, exc_value, exc_traceback):
                if issubclass(exc_type, KeyboardInterrupt):
                    sys.__excepthook__(exc_type, exc_value, exc_traceback)
                    return
                self.logger.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

            sys.excepthook = handle_exception


    def log_config(self, config, log_head=''):
        max_key_length = 10
        log_content = f'[{log_head}]\t'
        key_value_pairs = [f'{key}: {value}' for key, value in config.items()]
        log_content += ',\t'.join(key_value_pairs)

        self.log(log_content)
        self.log('-'*98)

    def exception(self, message, exc_info=True):
        self.logger.exception(message, exc_info=exc_info)

    def rename_log(self, result_dict):
        """Rename the .log file by prepending key result metrics to the filename."""

        rel_str, prepend_str = '', ''

        # Check if specific keywords are in result_dict and prepare them to be prepended
        prefered_metrics = ['OU-ave', 'ave', 'validity', 'NSPDK', 'FCD']
        for metric in result_dict:
            if metric in prefered_metrics:
                prepend_str += f'{metric}={result_dict[metric]:.6f}, '
            elif metric == 'sample_time':
                prepend_str += f'{metric}={result_dict[metric]}, '

        new_log_name = prepend_str + self.log_name
        self._rename_log(new_log_name)

    def _rename_log(self, new_name):
        # 1. Close the existing file handler.
        self.logger.removeHandler(self.file_handler)
        self.file_handler.close()

        # 2. Rename the file on disk.
        self.log_name = new_name
        new_log_path = os.path.join(self.log_dir, self.log_name)
        os.rename(self.log_path, new_log_path)

        # 3. Create a new file handler pointing to the renamed file.
        self.log_path = new_log_path
        new_file_handler = logging.FileHandler(self.log_path)
        new_file_handler.setLevel(self.logger.level)

        new_file_handler.setFormatter(self.formatter)

        self.file_handler = new_file_handler
        self.logger.addHandler(self.file_handler)

    def log(self, str, verbose=True):
        """Write a message to the log file.

        Args:
            verbose: If True, also print the message to the console.
        """


        try:
            with open(self.log_path, 'a') as f:
                f.write(str + '\n')

        except Exception as e:
            print(e)


        if verbose:
            print(str)
