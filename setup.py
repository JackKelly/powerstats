from setuptools import setup, find_packages

setup(
    name = "powerstats",
    version = "0.1",
    packages = find_packages(),
    install_requires = ['matplotlib', 'numpy'],
    author = "Jack Kelly",
    author_email = "jack-list@xlk.org.uk",
    description = "Create simple statistics from power data",
    license = "MIT",
    keywords = "power statistics python",
    url = "https://github.com/JackKelly/powerstats/",
    download_url = "https://github.com/JackKelly/powerstats/tarball/master#egg=powerstats-dev",
    long_description = open('README.md').read()
)
