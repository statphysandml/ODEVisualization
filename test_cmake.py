#!/usr/bin/env python3

import sys
import subprocess
import tempfile
from pathlib import Path

def test_cmake():
    """Test CMake configuration directly."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        build_temp = Path(temp_dir)
        
        cmake_args = [
            'cmake', 
            '/home/lukas/Repos/ODEVisualization',
            '-DPYTHON_EXECUTABLE=/home/lukas/.miniconda3/envs/frg/bin/python',
            '-DCMAKE_BUILD_TYPE=Release',
            '-DBUILD_PYTHON_BINDINGS=ON',
            '-DBUILD_DOCS=OFF',
            '-DBUILD_TESTING=OFF',
            '-DUSE_SYSTEM_DEVDAT=OFF',
            '-DUSE_SYSTEM_FLOWEQUATIONS=OFF',
            '-DUSE_SYSTEM_PARAMHELPER=OFF',
            '-DODEVIS_CMAKE_LIBRARY_OUTPUT_DIRECTORY=/tmp/test_output',
        ]
        
        print(f"Running CMake in temporary directory: {build_temp}")
        print(f"Command: {' '.join(cmake_args)}")
        
        try:
            result = subprocess.run(
                cmake_args,
                cwd=build_temp,
                capture_output=True,
                text=True,
                check=True
            )
            print("SUCCESS: CMake configuration completed")
            print("Last part of stdout:")
            print(result.stdout[-1000:])
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"FAILED: CMake configuration failed with return code {e.returncode}")
            print("STDOUT:")
            print(e.stdout[-2000:] if e.stdout else "No stdout")
            print("\nSTDERR:")
            print(e.stderr[-2000:] if e.stderr else "No stderr")
            return False

if __name__ == "__main__":
    success = test_cmake()
    sys.exit(0 if success else 1)
