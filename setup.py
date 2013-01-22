from setuptools import setup, find_packages

setup(
    name = "powerstats",
    version = "0.11",
    packages = find_packages(),
    install_requires = ['matplotlib', 'numpy', 'argparse'],
    author = "Jack Kelly",
    author_email = "",
    description = "Create simple statistics from power data",
    license = "MIT",
    keywords = "power statistics python",
    url = "https://github.com/JackKelly/powerstats/",
    download_url = "https://github.com/JackKelly/powerstats/tarball/master"
)
