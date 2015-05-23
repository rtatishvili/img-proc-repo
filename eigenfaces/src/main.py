#!/usr/bin/env python

import sys
import subprocess

# ...
# the rest of your module's code
# ...

if __name__ == '__main__':
    if '--unittest' in sys.argv:
        subprocess.call([sys.executable, '-m', 'unittest', 'discover'])