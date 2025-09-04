"""
{{ cookiecutter.project_slug.replace("-", " ").title() }} flow equations for ODEVisualization.
"""

from .{{ cookiecutter.project_slug.replace("-", "_") }} import {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}

__all__ = ['{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}']