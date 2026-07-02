from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

import sys
ext_modules = []
if sys.platform != "win32":
    ext_modules = [
        Pybind11Extension(
            "omni_bus_core",
            ["src/omni_bus_core.cpp"],
            libraries=["rt"] if "linux" in sys.platform else [],
        ),
    ]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="omnitrain",
    version="2.1.0",
    ext_modules=ext_modules,
    install_requires=["numpy", "torch", "rich", "pyyaml", "prompt_toolkit>=3.0", "pydantic", "pyserial"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    cmdclass={"build_ext": build_ext} if ext_modules else {},
)
