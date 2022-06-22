import os
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install


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

        cmake_args = ['-DBUILD_DOCS=OFF',
                      '-DBUILD_TESTING=OFF',
                      '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]
        if odevisualization_cmake_prefix_path is not None:
            cmake_args += ['-DCMAKE_PREFIX_PATH=' + odevisualization_cmake_prefix_path]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            cmake_args += ['-A', 'x64' if sys.maxsize > 2**32 else 'Win32']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j12']

        if gpu is not None:
            cmake_args += ['-DGPU=' + gpu]

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.', '--target', 'threepointsystem_python'] + build_args, cwd=self.build_temp)


class InstallCommand(install):
    user_options = install.user_options + [
        # ('someopt', None, None), # a 'flag' option
        ('odevisualization-cmake-prefix-path=', None, "CMAKE_PREFIX_PATH to the ODEVisualizationLib"), # an option that takes a value
        ('GPU', None, "Running on GPU")
    ]

    def initialize_options(self):
        install.initialize_options(self)
        # self.someopt = None
        self.odevisualization_cmake_prefix_path = None
        self.gpu = "ON"

    def finalize_options(self):
        #print("value of someopt is", self.someopt)
        print("odevisualization_cmake_prefix_path is", self.odevisualization_cmake_prefix_path)
        print("running on GPU", self.gpu)
        install.finalize_options(self)

    def run(self):
        global odevisualization_cmake_prefix_path
        if self.odevisualization_cmake_prefix_path is None:
            odevisualization_cmake_prefix_path = None
        else:
            odevisualization_cmake_prefix_path = self.odevisualization_cmake_prefix_path
        global gpu
        gpu = self.gpu
        install.run(self)


setup(
    name='Three Point System',
    version='0.0.1',
    author='Your Name',
    author_email='your@email.com',
    description='Add description here',
    long_description='',
    ext_modules=[CMakeExtension(
        name='threepointsystemsimulation'
    )],
    cmdclass=dict(build_ext=CMakeBuild, install=InstallCommand),
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    url='https://github.com/your_url',
    package_dir={"threepointsystem": "python_pybind"},
    packages=["threepointsystem"]
)
