from setuptools import find_packages, setup

setup(name='most-queue',
      version='1.62',
      description='Software package for calculation and simulation of queuing systems',
      author='Xabarov Roman',
      author_email='xabarov1985@gmail.com',
      url='https://github.com/xabarov/most-queue',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: >= 3.9',
          'Topic :: Text Processing :: Linguistic',
      ],
      license='MIT',
      packages=find_packages(),
      install_requires=[
          "colorama",
          "matplotlib",
          "numpy",
          "pandas",
          "scipy==1.13.0",
          "tqdm",
          "graphviz",
          "networkx",
          'pyyaml',
      ],
      include_package_data=True,
      zip_safe=False)
