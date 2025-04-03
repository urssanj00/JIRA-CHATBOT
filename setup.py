# setup.py
from setuptools import setup, find_packages

setup(
    name="jira-chatbot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    author="Sanjeev Kumar Pandey",
    author_email="csaa.sanjeev@gmial.com",
    description="A JIRA chatbot using NLP techniques",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/urssanj00/JIRA-CHATBOT/tree/DEV_1.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "jira-chatbot=src.main:main",
        ],
    },
)