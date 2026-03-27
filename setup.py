from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "omni_bus_core",
        ["src/omni_bus_core.cpp"],
        libraries=["rt"] if "linux" in str(__import__("sys").platform) else [],
    ),
]

setup(
    name="omnitrain",
    version="1.0.0",
    ext_modules=ext_modules,
    install_requires=["numpy", "torch", "rich", "pyyaml"],
    entry_points={
        'console_scripts': [
            'omni=cli:main',
        ],
    },
    cmdclass={"build_ext": build_ext},
)
