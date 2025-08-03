import sys
import os

import argparse

parser = argparse.ArgumentParser("Build Flow Equations")
parser.add_argument("-o", "--output-dir", help="Where to build the flow equations.", type=str)
# parser.add_argument("--config-file", help="User configuration file", type=str, default=None)
parser.add_argument("-n", "--project_name", help="The name of the flow equations.", type=str, default="My Flow Equation System")
parser.add_argument("-fep", "--flow_equation_path", help="Path to the flow_equations.txt and jacobian.txt files prepared for a parsing with the mathematicaparser and exported from Mathematica. If 'None', a default project with the flow equations of the Lorentz attractor is generated.", type=str, default="None")
parser.add_argument("-pyb", "--python_bindings", help="Whether building Python bindings should be support in the project: 'Yes' (default) or 'No'.", type=str, default="Yes")
parser.add_argument("-li", "--license", help="License: 'None', 'MIT', 'BSD-2', 'GPL-3.0', 'LGPL-3.0'.", type=str, default="None")


def generate_ode_system(output_dir, project_name, flow_equation_path="None", python_bindings="Yes", license="None"):
    from cookiecutter.main import cookiecutter
    cookiecutter(
        os.path.dirname(os.path.realpath(__file__)) + '/flow_equation_wrapper',
        no_input=True,
        output_dir=output_dir,
        extra_context={
            'project_name': project_name,
            'python_bindings': python_bindings,
            'license': license
        }
    )

    if flow_equation_path != "None":
        try:
            from mathematicaparser.odevisualization.generators import generate_equations
            import slugify
            theory_name = slugify.slugify(project_name).replace("-", "_")
            equation_target_dir = os.path.join(output_dir, slugify.slugify(project_name).replace("-", " ").title().replace(" ", ""))
            generate_equations(
                theory_name=theory_name,
                equation_path=flow_equation_path,
                project_path=equation_target_dir
            )
        except:
            print(sys.exc_info()[0])


if __name__ == "__main__":
    args = parser.parse_args()
    generate_ode_system(**vars(args))
