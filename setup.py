# setup.py
from setuptools import setup, find_packages

setup(
    name='datalib_ha',
    version='1.0.0',
    author='Ahmed Hamila',
    author_email='csahmed17@gmail.com',
    description='A comprehensive data analysis library for Python',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ahmedhamila/Datalib',
    project_urls={
        "Documentation": "https://datalib-ha.readthedocs.io",
        "Source Code": "https://github.com/ahmedhamila/datalib",
    },
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'scikit-learn>=0.24.0',
        'scipy>=1.7.0'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    keywords='data-analysis statistics machine-learning',
    python_requires='>=3.8',
    extras_require={
        'dev': ['pytest', 'sphinx']
    }
)