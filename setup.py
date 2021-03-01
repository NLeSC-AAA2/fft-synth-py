#!/usr/bin/env python

from setuptools import setup, find_packages  # type: ignore

test_deps = [
      "pytest>=5,<6",
      "pytest-cov>=2.8.1,<3",
      "pytest-mypy>=0.4.2,<1",
      "flake8>=3,<4",
      "pyopencl",
      "kernel_tuner[opencl]"]

setup(
    name="fftsynth",
    version="0.1.0",
    packages=find_packages(),

    install_requires=[
        "numpy",
        "jinja2"
    ],
    tests_require=test_deps,
    extras_require={
        "test": test_deps
    },
    include_package_data=True,
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.cl']
    },

    # metadata to display on PyPI
    author="Johan Hidding",
    author_email="j.hidding@esciencecenter.nl",
    description="FFT code generator for Intel FGPA OpenCL",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="fft",
    url="https://nlesc-aaa2.github.io/fftsynth-py",   # project home page, if any

    classifiers=[
          'License :: OSI Approved :: Apache Software License',
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Environment :: Console',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering',
          'Topic :: Software Development'
    ],

    # could also include long_description, download_url, etc.
)
