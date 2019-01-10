#!/usr/bin/env python
"""This script cleans the repository."""
import subprocess

subprocess.check_call(['git', 'clean', '-df'])
