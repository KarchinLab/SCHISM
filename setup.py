from setuptools import setup

setup(
    name='SCHISM',
    version='1.0.1',
    author='Noushin Niknafs',
    author_email='niknafs@jhu.edu',
    packages=['SCHISM'],
    scripts=['runSchism'],
    package_data={'':['*.yaml']},
    url='http://www.karchinlab.org/apps/appSchism.html',
    license='LICENSE.txt',
    description='SCHISM: Subclonal Hierarchy Inference from Somatic Mutations',
    long_description='\n' + open('README.txt').read(),
    install_requires=["numpy >= 1.7.1",\
                      "scipy >= 0.12.0",\
                      "PyYAML >= 3.11",\
                      "matplotlib >= 1.2.0"],
)

