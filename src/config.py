import os
import json
import logging
from pathlib import Path


def get_project_root() -> Path:
    """"""
    return Path(__file__).parent.parent


PROJECT_ROOT_DIR = get_project_root()

print(PROJECT_ROOT_DIR)

logging.basicConfig(
    # filename=os.path.join(PROJECT_ROOT_DIR, "data", "expert_bot_update.logs"),
    # filemode='a',
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', )

logger = logging.getLogger()
logger.setLevel(logging.INFO)