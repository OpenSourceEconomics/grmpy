#!/usr/bin/env python
"""This script updates the remote repository."""
import subprocess

subprocess.check_call(['git', 'commit', '-a', '-m"edits"'])
subprocess.check_call(['git', 'push'])
