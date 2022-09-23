from setuptools import setup, find_packages

setup(name='most-queue',
      version='0.11',
      description="Software package for calculation and simulation of queuing systems",
      author='Xabarov Roman',
      author_email='xabarov1985@gmail.com',
      url='https://github.com/xabarov/mps',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Topic :: Text Processing :: Linguistic',
      ],
      license='MIT',
      packages=find_packages(),
      install_requires=[
             "matplotlib>=3.5.2",
            "matplotlib-inline>=0.1.3",
            "numba>=0.56.0",
            "numpy>=1.22.4",
            "pandas>=1.4.3",
            "scipy>=1.9.0",
            "tqdm>=4.64.0",
            "tqdm-stubs>=0.2.1"
      ],
      include_package_data=True,
      zip_safe=False)