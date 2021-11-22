from os import path

PROJ_PATH = path.normpath(path.join(path.dirname(path.realpath(__file__))))
MODULE_PATH = path.normpath(path.join(path.dirname(path.realpath(__file__)),'..'))
DATA_PATH = path.join(MODULE_PATH,'data')

