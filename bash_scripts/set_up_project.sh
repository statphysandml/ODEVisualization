#!/bin/bash

# Project name
read -p "Enter project name (name of project directory): " project_name

if test -z $project_name; then
	echo "Project name cannot be empty."
	exit 1
fi

# Path to project root directory
read -e -p "Enter path to project generation directory (empty - project will be generated in default projects directory): " path_to_project_root

if test -z "$path_to_project_root"; then
	path_to_project_root="../projects"
fi

mkdir -p $path_to_project_root

if test -d "$path_to_project_root/$project_name"; then
	echo "Project already exists."
	exit 1
fi

# Path to ode_solver
read -e -p "Enter absolute path or relative path from project directory to program folder (empty - it is assumed path_to_program_directory='../../program/'): " path_to_program_directory
if test -z "$path_to_program_directory"; then
	path_to_program_directory="../../program"
fi
echo "CMakeLists.txt file will be written with path_to_program_directory=$path_to_program_directory"

# Create if default_equation_pickup directory does not exist yet
if ! test -d "$path_to_project_root/default_ode_pickup_location"; then
	mkdir "$path_to_project_root/default_ode_pickup_location"
	cp $path_to_project_root/default_ode_pickup_location/$path_to_program_directory/flow_equations/lorentz_attractor/flow_equations.txt $path_to_project_root/default_ode_pickup_location
	cp $path_to_project_root/default_ode_pickup_location/$path_to_program_directory/flow_equations/lorentz_attractor/jacobian.txt $path_to_project_root/default_ode_pickup_location
fi


# Generate folder structure
echo "Project is generated in $path_to_project_root/$project_name"
mkdir "$path_to_project_root/$project_name"
mkdir "$path_to_project_root/$project_name/cmake"
mkdir "$path_to_project_root/$project_name/data"
mkdir "$path_to_project_root/$project_name/run"
mkdir "$path_to_project_root/$project_name/flow_equations"
mkdir "$path_to_project_root/$project_name/bash_scripts"

# mkdir "$path_to_project_root/$project_name/debug"
# mkdir "$path_to_project_root/$project_name/gpu_debug"
# mkdir "$path_to_project_root/$project_name/gpu_release"

# Generate CMakeLists.txt file
source write_cmake_lists_file.sh

# Generate main.cu file
source write_main_cu_file.sh

# Add bash bash_scripts
source write_add_or_overwrite_theory_bash_script.sh
