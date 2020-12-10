cat >$path_to_project_root/$project_name/bash_scripts/add_or_overwrite_theory.sh <<EOL
#!/bin/bash

read -e -p "Enter path to flow_equations.txt and jacobian.txt file (empty - equations are taken from the folder default_ode_pickup_location): " path_to_source_files

if test -z "\$path_to_source_files"; then
	path_to_source_files="../../default_ode_pickup_location"
fi

read -p "Enter theory name: " theory_name

if test -z \$theory_name; then
	echo "Theory name cannot be empty."
	exit 1
fi

flow_equation_path="../flow_equations/\$theory_name"

mkdir -p \$flow_equation_path

# Copy files
cp \$path_to_source_files/flow_equations.txt \$flow_equation_path
cp \$path_to_source_files/jacobian.txt \$flow_equation_path

source activate flowequation
python ../$path_to_program_directory/mathematica_parser/main.py \$theory_name 10 40 ../ ../${path_to_program_directory}

mkdir -p "$path_to_project_root/$project_name/data/\$theory_name"

cd ../run/
cmake ../cmake/ -DCMAKE_BUILD_TYPE=Release
make -j6

EOL
