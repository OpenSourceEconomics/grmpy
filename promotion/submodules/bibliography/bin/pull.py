#!/usr/bin/env python
"""This script updates the local repository."""
import subprocess

subprocess.check_call(['git', 'pull'])
