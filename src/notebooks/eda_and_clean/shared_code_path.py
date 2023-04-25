

#https://stackoverflow.com/questions/34478398/import-local-function-from-a-module-housed-in-another-directory-with-relative-im


import sys
import os

module_path = os.path.abspath(os.path.join("..", "..", "shared_code"))
if module_path not in sys.path:
    sys.path.append(module_path)
