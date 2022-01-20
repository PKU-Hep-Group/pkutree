import setuptools

setuptools.setup(
    name='pkutree', 
    version='0.1.0',
    description='pkutree test',
    author='jixiao',
    author_email='jie.xiao@cern.ch',
    url='https://github.com/PKU-Hep-Group/pkutree',
    packages=setuptools.find_packages(),
    package_data={
        "pkutree": [
            "config/*yaml",
            "data/btagSF/2016/*",
            "data/btagSF/2017/*",
            "data/btagSF/2018/*",
        ]
    }
)
