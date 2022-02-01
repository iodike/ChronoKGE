"""
Main
"""

#import os

from chrono_kge.main.manager.run_manager import RunManager
from chrono_kge.main.manager.tune_manager import TuneManager
from chrono_kge.utils.vars.constants import COMMAND
from chrono_kge.utils.logger import logger
from chrono_kge.main.manager.config_manager import ConfigManager
from chrono_kge.main.parser.parser import UIParser


'''Disables asynchronous kernel launches / CUDA debugging'''
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


'''MAIN'''

if __name__ == '__main__':

    '''Parse arguments'''
    ui_parser = UIParser()
    args = ui_parser.parser.parse_args()

    '''Init framework'''
    config = ConfigManager()
    config.init()

    '''Init manager'''
    manager = None

    '''Switch command'''
    if args.command == COMMAND.RUN:
        manager = RunManager(vars(args))

    elif args.command == COMMAND.TUNE:
        manager = TuneManager(vars(args))

    else:
        logger.warning("Unknown command.")
        ui_parser.parser.print_help()
        exit(1)

    '''Start runner'''
    manager.setup()

    '''Exit'''
    logger.info("Finished.")
    exit(0)
