# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 03:20:01 2021

@author: eduar
"""

import os
import urllib.request


def load_file(local_file, file, url=None, force=False):
    """
    Check file or store it locally
    """
    if not os.path.exists(os.path.join(local_file, file)) or force:
      if url:
        print('Downloading')
        with urllib.request.urlopen(url) as opener, \
          open(os.path.join(local_file, file), mode='w', encoding='utf8') as outfile:
            outfile.write(opener.read())
      else: 
        print('File not found')
        raise OSError(-1, "File not Found")
    else:
        print('File already exists')