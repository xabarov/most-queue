from setuptools import setup, find_packages

setup(name='most-queue',
      version='1.45',
      description='Software package for calculation and simulation of queuing systems',
      author='Xabarov Roman',
      author_email='xabarov1985@gmail.com',
      url='https://github.com/xabarov/most-queue',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.9',
          'Topic :: Text Processing :: Linguistic',
      ],
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'matplotlib',
          'numpy',
          'pandas',
          'scipy',
          'tqdm',
          'colorama',
      ],
      include_package_data=True,
      zip_safe=False)
