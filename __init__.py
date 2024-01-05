import os
import sys

import threestudio
from packaging.version import Version

sys.path.append(os.path.abspath(os.path.join("custom", "threestudio-3dgs")))

if hasattr(threestudio, "__version__") and Version(threestudio.__version__) >= Version(
    "0.2.1"
):
    pass
else:
    if hasattr(threestudio, "__version__"):
        print(f"[INFO] threestudio version: {threestudio.__version__}")
    raise ValueError(
        "threestudio version must be >= 0.2.0, please update threestudio by pulling the latest version from github"
    )

import gsstudio
