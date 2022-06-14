from setuptools import setup


setup(
    name="lxd-exporter",
    version="0.4.8",
    include_package_data=True,
    license="Apache Software License",
    description='prometheus exporter for LXD clusters',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Dmitry Orlov",
    author_email="me@mosquito.su",
    url="https://github.com/mosquito/lxd-exporter",
    project_urls={
        "Source": "https://github.com/mosquito/lxd-exporter/",
        "Tracker": "https://github.com/mosquito/lxd-exporter/issues",
        "Say Thanks!": "https://saythanks.io/to/mosquito",
    },
    packages=["."],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
        "Operating System :: Microsoft",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    python_requires=">=3.8.*, <4",
    install_requires=[
        "aiomisc~=16.0",
        "aiohttp~=3.8",
        "argclass~=0.8",
    ],
    entry_points={
        "console_scripts": ["lxd-exporter = lxd_exporter:main"]
    },
)
