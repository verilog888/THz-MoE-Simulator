import pathlib
import subprocess
import sysconfig
import pybind11
root = pathlib.Path(__file__).resolve().parents[1]
subprocess.run(['g++','-O3','-g0','-std=c++20','-fPIC','-shared','-Wall','-Wextra',
                '-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),
                str(root/'vdie_moe_sim/native/core.cpp'),'-o',
                str(root/('vdie_moe_sim/_native'+sysconfig.get_config_var('EXT_SUFFIX')))],
               check=True)
