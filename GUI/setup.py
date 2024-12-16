#!/usr/bin/env python3

from xasyversion import version
from setuptools import setup

setup(
    name="xasy",
    version=version.VERSION,
    author="Supakorn Rassameemasmuang, Orest Shardt, and John C. Bowman",
    description="User interface for Asymptote, a vector graphics language",
    url="https://asymptote.sourceforge.io",
    download_url="https://sourceforge.net/projects/asymptote/"
)
