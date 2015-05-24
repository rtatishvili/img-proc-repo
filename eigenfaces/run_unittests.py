#!/usr/bin/env python

import sys
import subprocess

if __name__ == '__main__':
    subprocess.call([sys.executable, '-m', 'unittest', 'discover'])
