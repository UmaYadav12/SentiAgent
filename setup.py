from setuptools import find_packages, setup

setup(
    name="sentiagent",
    version="1.0.0",
    description=(
        "SentiAgent: An Agentic LLM Framework for Context-Aware and "
        "Explainable Sentiment Analysis"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "examples"]),
    python_requires=">=3.9",
    install_requires=[
        "requests>=2.31.0",
        "anthropic>=0.40.0",
    ],
    extras_require={
        "openai": ["openai>=1.40.0"],
        "dev": ["pytest>=7.4.0"],
    },
    entry_points={
        "console_scripts": [
            "sentiagent=sentiagent.cli:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
