from setuptools import setup


setup(
    name="lxd-exporter",
    version="0.3.6",
    include_package_data=True,
    license="Apache Software License",
    author="Dmitry Orlov",
    author_email="me@mosquito.su",
    url="https://github.com/mosquito/lxd-exporter",
    project_urls={
        "Source": "https://github.com/mosquito/lxd-exporter/",
        "Tracker": "https://github.com/mosquito/lxd-exporter/issues",
        "Say Thanks!": "https://saythanks.io/to/me%40mosquito.su",
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
        "Flask==1.1.2",
        "gevent==21.1.2",
        "gunicorn~=20.1.0",
        "itsdangerous==1.1.0",
        "Jinja2==2.11.3",
        "MarkupSafe==1.1.1",
        "prometheus-client==0.10.0",
        "pylxd==2.3.0",
        "werkzeug==1.0.1",
    ],
    entry_points={
        "console_scripts": ["lxd-exporter = lxd_exporter:main"]
    },
)
