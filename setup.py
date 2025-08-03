import os
import sys
import platform
import subprocess
import multiprocessing
from pathlib import Path
import shutil
import tempfile
from typing import List, Optional

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.develop import develop


class CMakeExtension(Extension):
    """A CMake extension that will be compiled using CMake."""
    
    def __init__(self, name: str, sourcedir: str = '', cmake_args: Optional[List[str]] = None):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.cmake_args = cmake_args or []


class CMakeBuild(build_ext):
    """Custom build extension for CMake projects."""
    
    def run(self):
        """Run the build process."""
        self.check_cmake()
        for ext in self.extensions:
            self.build_extension(ext)

    def check_cmake(self):
        """Check if CMake is available."""
        try:
            subprocess.check_output(['cmake', '--version'])
        except (OSError, subprocess.CalledProcessError):
            raise RuntimeError(
                "CMake must be installed to build this package. "
                "Please install CMake 3.18 or higher."
            )

    def build_extension(self, ext: CMakeExtension):
        """Build a single CMake extension."""
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return
            
        # Get the directory where the compiled extension will be placed
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # Create build directory
        build_temp = Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)
        
        # Detect build configuration
        cfg = 'Debug' if self.debug else 'Release'
        
        # Determine if we should use superbuild or direct build
        use_superbuild = self._should_use_superbuild()
        
        if use_superbuild:
            self._build_with_superbuild(ext, extdir, cfg, build_temp)
        else:
            self._build_direct(ext, extdir, cfg, build_temp)

    def _should_use_superbuild(self) -> bool:
        """Determine whether to use superbuild or direct build."""
        # Check for environment variable override
        if 'ODEVIS_USE_SUPERBUILD' in os.environ:
            return os.environ['ODEVIS_USE_SUPERBUILD'].lower() in ('1', 'true', 'on', 'yes')
        
        # Check if dependencies are available in the system
        try:
            import subprocess
            # Try to find the dependencies using pkg-config or cmake
            for dep in ['devdat', 'flowequations', 'paramhelper']:
                result = subprocess.run(
                    ['pkg-config', '--exists', dep], 
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    # If any dependency is missing, use superbuild
                    return True
            return False
        except FileNotFoundError:
            # If pkg-config is not available, default to superbuild
            return True

    def _build_with_superbuild(self, ext: CMakeExtension, extdir: str, cfg: str, build_temp: Path):
        """Build using the superbuild approach."""
        print("Building with superbuild (automatic dependency management)...")
        
        # Use the current directory as the superbuild root
        superbuild_dir = Path(ext.sourcedir).resolve()
        
        cmake_args = self._get_base_cmake_args(extdir, cfg)
        cmake_args.extend([
            '-DBUILD_PYTHON_BINDINGS=ON',
            '-DBUILD_DOCS=OFF',
            '-DBUILD_TESTING=OFF',
            '-DUSE_SYSTEM_DEVDAT=OFF',
            '-DUSE_SYSTEM_FLOWEQUATIONS=OFF',
            '-DUSE_SYSTEM_PARAMHELPER=OFF',
            '-DUSE_SYSTEM_PYBIND11=ON',  # Use system pybind11 to avoid version issues
            # Add policy version minimum to handle old dependencies
            '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
            # Pass the library output directory for the odevis subproject
            f'-DODEVIS_CMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
        ])
        
        # Add any additional cmake args from the extension
        cmake_args.extend(ext.cmake_args)
        
        build_args = self._get_build_args(cfg)
        
        # Configure and build
        self._run_cmake_configure(superbuild_dir, cmake_args, build_temp)
        self._run_cmake_build(build_args, build_temp)
        
        # Copy the built Python module
        self._copy_built_extension(build_temp, extdir, 'odevisualizationlib')

    def _build_direct(self, ext: CMakeExtension, extdir: str, cfg: str, build_temp: Path):
        """Build directly against system dependencies."""
        print("Building directly against system dependencies...")
        
        # Use the odevis subdirectory for direct build
        odevis_dir = Path(ext.sourcedir) / "odevis"
        
        cmake_args = self._get_base_cmake_args_direct(extdir, cfg)
        cmake_args.extend([
            '-DBUILD_PYTHON_BINDINGS=ON',
            '-DBUILD_DOCS=OFF',
            '-DBUILD_TESTING=OFF',
            '-DUSE_SYSTEM_PYBIND11=ON',  # Use system pybind11 to avoid version issues
        ])        # Add any additional cmake args from the extension
        cmake_args.extend(ext.cmake_args)
        
        build_args = self._get_build_args(cfg)
        
        # Configure and build
        self._run_cmake_configure(odevis_dir, cmake_args, build_temp)
        self._run_cmake_build(build_args, build_temp, target='odevisualization_python')
        
        # Copy the built Python module
        self._copy_built_extension(build_temp, extdir, 'odevisualizationlib')

    def _get_base_cmake_args(self, extdir: str, cfg: str) -> List[str]:
        """Get base CMake arguments."""
        # Create a temporary install directory to avoid permission issues
        install_dir = Path(extdir).parent / "cmake_install"
        install_dir.mkdir(exist_ok=True)
        
        cmake_args = [
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE={cfg}',
            f'-DCMAKE_INSTALL_PREFIX={install_dir}',
        ]
        
        # Add CUDA architectures if specified
        if hasattr(self, 'cmake_cuda_architectures') and self.cmake_cuda_architectures:
            cmake_args.append(f'-DCMAKE_CUDA_ARCHITECTURES={self.cmake_cuda_architectures}')
        
        # Platform-specific arguments
        if platform.system() == "Windows":
            cmake_args.extend([
                f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}',
                '-A', 'x64' if sys.maxsize > 2**32 else 'Win32'
            ])
        
        return cmake_args

    def _get_base_cmake_args_direct(self, extdir: str, cfg: str) -> List[str]:
        """Get base CMake arguments for direct build."""
        # Create a temporary install directory to avoid permission issues
        install_dir = Path(extdir).parent / "cmake_install"
        install_dir.mkdir(exist_ok=True)
        
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE={cfg}',
            f'-DCMAKE_INSTALL_PREFIX={install_dir}',
        ]
        
        # Add CUDA architectures if specified
        if hasattr(self, 'cmake_cuda_architectures') and self.cmake_cuda_architectures:
            cmake_args.append(f'-DCMAKE_CUDA_ARCHITECTURES={self.cmake_cuda_architectures}')
        
        # Platform-specific arguments
        if platform.system() == "Windows":
            cmake_args.extend([
                f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}',
                '-A', 'x64' if sys.maxsize > 2**32 else 'Win32'
            ])
        
        return cmake_args

    def _get_build_args(self, cfg: str) -> List[str]:
        """Get build arguments."""
        build_args = ['--config', cfg]
        
        if platform.system() == "Windows":
            build_args.extend(['--', '/m'])
        else:
            # Use all available cores, but cap at 12 to avoid memory issues
            num_cores = min(multiprocessing.cpu_count(), 12)
            build_args.extend(['--', f'-j{num_cores}'])
        
        return build_args

    def _run_cmake_configure(self, source_dir: str, cmake_args: List[str], build_temp: Path):
        """Run CMake configure step."""
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version()
        )
        
        configure_cmd = ['cmake', str(source_dir)] + cmake_args
        print(f"Configuring with: {' '.join(configure_cmd)}")
        print(f"Working directory: {build_temp}")
        print(f"Source directory: {source_dir}")
        print(f"Build temp exists: {build_temp.exists()}")
        
        # Try to run the command and capture both stdout and stderr
        try:
            result = subprocess.run(
                configure_cmd, 
                cwd=build_temp, 
                env=env, 
                capture_output=True, 
                text=True,
                check=True
            )
            print("CMake configure completed successfully")
            if result.stdout:
                print("STDOUT:", result.stdout[-1000:])  # Last 1000 chars
        except subprocess.CalledProcessError as e:
            print(f"CMake configure failed with return code {e.returncode}")
            print("STDOUT:", e.stdout[-1000:] if e.stdout else "No stdout")
            print("STDERR:", e.stderr[-1000:] if e.stderr else "No stderr")
            raise

    def _run_cmake_build(self, build_args: List[str], build_temp: Path, target: Optional[str] = None):
        """Run CMake build step."""
        build_cmd = ['cmake', '--build', '.']
        if target:
            build_cmd.extend(['--target', target])
        build_cmd.extend(build_args)
        
        print(f"Building with: {' '.join(build_cmd)}")
        subprocess.check_call(build_cmd, cwd=build_temp)

    def _copy_built_extension(self, build_temp: Path, extdir: str, module_name: str):
        """Copy the built extension to the correct location."""
        
        # First check if the module is already in the target directory
        extdir_path = Path(extdir)
        existing_modules = list(extdir_path.glob(f'{module_name}.*'))
        for module in existing_modules:
            if module.suffix in ['.so', '.dll', '.dylib', '.pyd'] and not module.is_symlink():
                print(f"Module {module_name} already exists in target directory: {module}")
                return
        
        # Look for the built module in various possible locations
        possible_locations = [
            build_temp / 'odevis' / 'python' / f'{module_name}.*',
            build_temp / 'python' / f'{module_name}.*',
            build_temp / f'{module_name}.*',
            build_temp / 'install' / 'lib' / f'{module_name}.*',
            build_temp / 'odevisualization_main-prefix' / 'src' / 'odevisualization_main-build' / 'python' / f'{module_name}.*',
        ]
        
        built_module = None
        for pattern in possible_locations:
            if pattern.parent.exists():
                matches = list(pattern.parent.glob(pattern.name))
                if matches:
                    # Find the actual shared library (not symlinks)
                    for match in matches:
                        if match.suffix in ['.so', '.dll', '.dylib', '.pyd'] and not match.is_symlink():
                            built_module = match
                            break
                    if built_module:
                        break
        
        if not built_module:
            # Print available files for debugging
            print("Available files in build directory:")
            for root, dirs, files in os.walk(build_temp):
                for file in files:
                    if any(ext in file for ext in ['.so', '.dll', '.dylib', '.pyd']):
                        print(f"  {os.path.join(root, file)}")
            raise RuntimeError(f"Could not find built module '{module_name}' in build directory")
        
        # Copy to the extension directory
        dest = Path(extdir) / built_module.name
        print(f"Copying {built_module} to {dest}")
        shutil.copy2(built_module, dest)


class CustomInstallCommand(install):
    """Custom install command with additional options."""
    
    user_options = install.user_options + [
        ('cmake-cuda-architectures=', None, "CUDA architectures to build for"),
        ('use-superbuild', None, "Force use of superbuild"),
        ('use-system-deps', None, "Force use of system dependencies"),
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.cmake_cuda_architectures = None
        self.use_superbuild = None
        self.use_system_deps = None

    def finalize_options(self):
        install.finalize_options(self)
        if self.cmake_cuda_architectures:
            print(f"CUDA architectures: {self.cmake_cuda_architectures}")
        
        # Set environment variables for build process
        if self.use_superbuild:
            os.environ['ODEVIS_USE_SUPERBUILD'] = '1'
        elif self.use_system_deps:
            os.environ['ODEVIS_USE_SUPERBUILD'] = '0'

    def run(self):
        # Pass cmake_cuda_architectures to build_ext
        if self.cmake_cuda_architectures:
            for cmd in self.distribution.command_obj.values():
                if hasattr(cmd, 'cmake_cuda_architectures'):
                    cmd.cmake_cuda_architectures = self.cmake_cuda_architectures
        
        install.run(self)


class CustomDevelopCommand(develop):
    """Custom develop command for development installs."""
    
    user_options = develop.user_options + [
        ('cmake-cuda-architectures=', None, "CUDA architectures to build for"),
    ]

    def initialize_options(self):
        develop.initialize_options(self)
        self.cmake_cuda_architectures = None

    def finalize_options(self):
        develop.finalize_options(self)

    def run(self):
        if self.cmake_cuda_architectures:
            for cmd in self.distribution.command_obj.values():
                if hasattr(cmd, 'cmake_cuda_architectures'):
                    cmd.cmake_cuda_architectures = self.cmake_cuda_architectures
        
        develop.run(self)

if __name__ == "__main__":
    setup(
        ext_modules=[CMakeExtension('odevisualizationlib')],
        cmdclass={
            'build_ext': CMakeBuild,
            'install': CustomInstallCommand,
            'develop': CustomDevelopCommand,
        },
        zip_safe=False,
    )
