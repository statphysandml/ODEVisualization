cat >$path_to_project_root/$project_name/main.cu <<EOL
#include "$path_to_program_directory/ode_solver/include/executer.hpp"

void custom_main();

int main(int argc, char **argv)
{
    Py_Initialize();
    PyRun_SimpleString("import sys\n" "import os");
    PyRun_SimpleString("sys.path.append( os.path.dirname(os.getcwd()) + '/../../program/')");
    PyRun_SimpleString("sys.path.append( os.path.dirname(os.getcwd()) + '/../../program/plotting_routines/')");
    PyRun_SimpleString("sys.path.append( os.path.dirname(os.getcwd()) + '/../../program/plotting_routines/plotting_environment/')");
    PyRun_SimpleString("print('Running python in ' + os.path.dirname(os.getcwd()) + '/../../program/plotting_routines/')");

    std::cout << gcp() << std::endl;

    if(argc > 1)
        run_from_file(argc, argv);
    else
        custom_main();

    Py_Finalize();
}

void custom_main()
{}

EOL
