# This script executes before the project is generated from your cookiecutter.
# Details about hooks can be found in the cookiecutter documentation:
# https://cookiecutter.readthedocs.io/en/latest/advanced/hooks.html
#
# An example of a pre-hook would be to validate the provided input for a
# user configuration value and exit with an error upon failure.

import sys


def fail_if(condition, message):
    if condition:
        sys.stderr.write(message + "\n")
        sys.exit(1)

# fail_if(
#     "{{ cookiecutter.python_bindings }}" == "No",
#     "Can't do PyPI release without building Python bindings"
# )
