from setuptools import setup, find_packages

setup(
    name='aiga',
    version='0.1',
    description='Exploritary Module for Machine Learning',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author='Joshua Windle',
    author_email='Joshua.W.Windle@gmail.com',
    url="",
    license="GPL-3.0",
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy'
    ],
    extras_require={
    },
    entry_points={
        "console_scripts": [
            "aiga=aiga.main:main_function"
        ]
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.9', 
    install_requires=['numpy',
                      'scipy',
                      'matplotlib'],
    project_urls={
        'Acknowledgements': 'https://openai.com/chatgpt',
    }
)
