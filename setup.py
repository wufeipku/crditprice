# ============================================================================
# @Time :  
# @Author: Wufei
# @File: setup.py.py
# ============================================================================
# -*- coding: utf-8 -*-

import setuptools

with open("README", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
     name="creditprice",
     version="0.0.8",
     author="Fei Wu",
     author_email="wufei.pku@163.com",
     description="Credit price tool",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/pypa/sampleproject",
     packages=setuptools.find_packages(),
     install_requires=[
         'pandas>=0.24.0',
         'numpy>=1.16.0',
         'scipy>=1.4.1',
         'joblib>=0.14.0',
         'matplotlib'
     ],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
)