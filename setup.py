from setuptools import setup, find_packages

setup(
    name='PythonAQ',
    version='0.1.0',
    description='Air Quality Data Processing and Visualization Toolkit',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Chris Rushton',
    author_email='c.e.rushton@leeds.ac.uk',
    url='https://github.com/yourusername/airpy',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'pandas==2.2.2'
        'numpy==1.26.0'
        'requests==2.31.0'
        'rdata==0.11.2'
        'plotly==5.22.0'
        'scikit-learn==1.5.1'
        'pygam==0.9.0'
        'scipy==1.11.0'
        'streamlit==1.38.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)