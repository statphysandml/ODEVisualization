# This script executes after the project is generated from your cookiecutter.
# Details about hooks can be found in the cookiecutter documentation:
# https://cookiecutter.readthedocs.io/en/latest/advanced/hooks.html
#
# An example of a post-hook would be to remove parts of the project
# directory tree based on some configuration values.

import os
import subprocess
import sys
from cookiecutter.utils import rmtree


# Optionally remove files whose existence is tied to disabled features
def conditional_remove(condition, path):
    if condition:
        if os.path.isfile(path):
            os.remove(path)
        else:
            rmtree(path)

conditional_remove("{{ cookiecutter.license }}" == "None", "LICENSE.md")
conditional_remove("{{ cookiecutter.python_bindings }}" == "No", "setup.py")
conditional_remove("{{ cookiecutter.python_bindings }}" == "No", "python_pybind")
conditional_remove("{{ cookiecutter.python_bindings }}" == "No", "python_examples/mode_simulation.py")

# Print a message about success
print("The project {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }} was successfully generated!")
print("The file README.md contains instructions for installing and integrating the newly generated flow equation system.")
