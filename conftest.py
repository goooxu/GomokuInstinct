"""让 pytest 无论从哪个目录启动都能 import 到仓库内的包。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
