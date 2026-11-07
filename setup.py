"""Legacy installer entry point for older offline Python environments."""

from setuptools import find_packages, setup

setup(
    name="prism-p2",
    version="0.1.0",
    description="Physics-guided Reconstruction Inspection and Sensor Monitoring",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "prism-inventory=prism.data.inventory:main",
            "prism-smoke=prism.smoke:main",
        ]
    },
)
