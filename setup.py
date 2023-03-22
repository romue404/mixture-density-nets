from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(name='Mixture-Density-Nets',
      version='0.1.0',
      description='A small PyTorch library for Mixture Density Networks.',
      author='Robert Müller',
      author_email='robert.mueller1990@googlemail.com',
      url='https://github.com/romue404/mixture-density-nets',
      license='MIT',
      keywords=[
            'artificial intelligence',
            'pytorch',
            'mixture density network'
      ],
      classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.6',
      ],
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(exclude=['examples']),
      install_requires=['torch>=1.6', 'numpy']
      )
