import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.readlines()

setuptools.setup(
    name="ydocr",
    version="1.2",
    author="jiaer",
    author_email="jia.er@winrobot360.com",
    license='Apache 2.0',
    description="影刀离线OCR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mingo-doer/ydocr",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
