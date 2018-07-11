#!/usr/bin/env python3

import gettext

p = property

class xasyString:
    def __init__(self, lang=None):
        s = self
        if lang is None:
            _ = lambda x:  x 
        else:
            lng = gettext.translation('base', localedir='GUI/locale', languages=[lang])
            lng.install()
            _ = lng.gettext
            
        s.rotate = _('Rotate')
        s.scale = _('Scale')
        s.translate = _('Translate')

        s.fileOpenFailed = _('File Opening Failed.')
        s.fileOpenFailedText = _('File could not be opened.')
        s.asyfyComplete = _('Ready.')
