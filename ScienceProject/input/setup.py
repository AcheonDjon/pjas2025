# filepath: /c:/Users/manoj/Downloads/ScienceProject/setup.py
from setuptools import setup, find_packages

setup(
    name='ScienceProject',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'opencv-python',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for tracking particles in videos',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/ScienceProject',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)