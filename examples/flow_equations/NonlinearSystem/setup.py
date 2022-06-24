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

        cmake_args = ['-DBUILD_DOCS=OFF',
                      '-DBUILD_TESTING=OFF',
                      '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]
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
        subprocess.check_call(['cmake', '--build', '.', '--target', 'nonlinearsystem_python'] + build_args, cwd=self.build_temp)


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
    name='Nonlinear System',
    version='0.0.1',
    author='Your Name',
    author_email='your@email.com',
    description='Add description here',
    long_description='',
    ext_modules=[CMakeExtension(
        name='nonlinearsystemsimulation'
    )],
    cmdclass=dict(build_ext=CMakeBuild, install=InstallCommand),
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    url='https://github.com/your_url',
    package_dir={"nonlinearsystem": "python_pybind"},
    packages=["nonlinearsystem"]
)
