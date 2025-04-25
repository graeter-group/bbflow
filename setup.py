from setuptools import setup

setup(
    name="bbflow",
    packages=[
        "bbflow",
    ],
    package_dir={
        "bbflow": './bbflow',
    },
    entry_points={
        "console_scripts": [
            "bbflow_sample=bbflow.deployment.cmd_line_entry:_sample",
        ],
    },
    tests_require=["pytest"],
)