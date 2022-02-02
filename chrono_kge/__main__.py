"""
Main
"""

#import os

from chrono_kge.main.manager.run_manager import RunManager
from chrono_kge.main.manager.tune_manager import TuneManager
from chrono_kge.utils.vars.constants import COMMAND
from chrono_kge.utils.logger import logger
from chrono_kge.main.manager.config_manager import ConfigManager
from chrono_kge.main.parser.parser_cmd import CMD_Parser
from chrono_kge.main.parser.parser_yaml import YML_Parser


'''Disables asynchronous kernel launches / CUDA debugging'''
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


'''MAIN'''

if __name__ == '__main__':

    cmd_parser = CMD_Parser()
    yml_parser = YML_Parser()

    '''Parse arguments'''
    args = cmd_parser.parser.parse_args()

    '''Init framework'''
    config = ConfigManager()
    config.init()

    '''Init manager'''
    manager = None

    '''Switch command'''
    if args.command == COMMAND.RUN:
        if args.main_config != "" and args.run_config != "":
            main_args = yml_parser.load_args(args.main_config)
            run_args = yml_parser.load_args(args.run_config)
            manager = RunManager({**main_args['hp'], **run_args['hp']})
        else:
            manager = RunManager(vars(args))

    elif args.command == COMMAND.TUNE:
        manager = TuneManager(vars(args))

    else:
        logger.warning("Unknown command.")
        cmd_parser.parser.print_help()
        exit(1)

    '''Start runner'''
    manager.setup()

    '''Exit'''
    logger.info("Finished.")
    exit(0)
