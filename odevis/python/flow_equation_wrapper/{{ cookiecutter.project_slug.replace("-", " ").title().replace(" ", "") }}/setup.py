import os
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install


cmake_cuda_architectures = None


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # Find the ODEVisualization superbuild install directory
        odevis_install_dir = None
        
        # Method 1: Try to use environment variable if set
        if 'ODEVISUALIZATION_INSTALL_DIR' in os.environ:
            odevis_install_dir = os.environ['ODEVISUALIZATION_INSTALL_DIR']
            if not os.path.exists(odevis_install_dir):
                print(f"Warning: ODEVISUALIZATION_INSTALL_DIR set to {odevis_install_dir} but directory doesn't exist")
                odevis_install_dir = None
        
        # Method 2: Try to import odesolver and find path
        if odevis_install_dir is None:
            try:
                import odesolver
                odesolver_path = os.path.dirname(odesolver.__file__)
                # The odesolver package is in odevis/python/odesolver, so go up to find cmake_install
                # odesolver_path is typically: /path/to/ODEVisualization/odevis/python/odesolver
                odevis_dir = os.path.dirname(os.path.dirname(odesolver_path))  # Go up two levels to odevis
                potential_install_dir = os.path.join(odevis_dir, 'cmake_install')
                if os.path.exists(potential_install_dir):
                    odevis_install_dir = potential_install_dir
            except ImportError:
                pass  # Will try other methods
        
        # Method 3: Try relative path from current location (fallback)
        if odevis_install_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Look for the odevis/cmake_install directory relative to this package
            repo_root = os.path.abspath(os.path.join(current_dir, '../../..'))
            potential_install_dir = os.path.join(repo_root, 'odevis', 'cmake_install')
            if os.path.exists(potential_install_dir):
                odevis_install_dir = potential_install_dir
        
        # If all methods failed, provide helpful error message
        if odevis_install_dir is None:
            raise RuntimeError(
                "Could not find ODEVisualization superbuild install directory. "
                "Please ensure ODEVisualization is installed first with 'pip install -e .' in the root directory, "
                "or set the ODEVISUALIZATION_INSTALL_DIR environment variable to the cmake_install directory."
            )

        cmake_args = ['-DBUILD_DOCS=OFF',
                      '-DBUILD_TESTING=OFF',
                      '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      f'-DSUPERBUILD_INSTALL_DIR={odevis_install_dir}']
        if cmake_cuda_architectures is not None:
            cmake_args += ['-DCMAKE_CUDA_ARCHITECTURES=' + cmake_cuda_architectures]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            cmake_args += ['-A', 'x64' if sys.maxsize > 2**32 else 'Win32']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j12']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.', '--target', '{{ cookiecutter.project_slug.replace("-", "") }}_python'] + build_args, cwd=self.build_temp)


class InstallCommand(install):
    user_options = install.user_options + [
        # ('someopt', None, None), # a 'flag' option
        ('cmake-cuda-architectures=', None, "CMAKE_CUDA_ARCHITECTURES"),
    ]

    def initialize_options(self):
        install.initialize_options(self)
        # self.someopt = None
        self.cmake_cuda_architectures = None

    def finalize_options(self):
        #print("value of someopt is", self.someopt)
        print("cmake_cuda_architectures", self.cmake_cuda_architectures)
        install.finalize_options(self)

    def run(self):
        global cmake_cuda_architectures
        if self.cmake_cuda_architectures is not None:
            cmake_cuda_architectures = self.cmake_cuda_architectures
        install.run(self)


setup(
    ext_modules=[CMakeExtension(
        name='{{ cookiecutter.project_slug.replace("-", "") }}simulation'
    )],
    cmdclass=dict(build_ext=CMakeBuild, install=InstallCommand),
    zip_safe=False,
)
