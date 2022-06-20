from odesolver.flow_equations import FlowEquations


class {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}(FlowEquations):
    def __init__(self):
        from {{ cookiecutter.project_slug.replace("-", "") }}simulation import {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}Flow, {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}Jacobians
        super().__init__(flow={{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}Flow(), jacobians={{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}Jacobians())
