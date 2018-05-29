import gettext

p = property

class xasyString:
    def __init__(self, lang=None):
        if lang is None:
            self._ = gettext.gettext
        else:
            lng = gettext.translation('base', localedir='GUI/locale', languages=[lang])
            lng.install()
            self._ = lng.gettext

    @p
    def rotate(s):
        return s._('Rotate')

    @p
    def scale(s):
        return s._('Scale')

    @p
    def translate(s):
        return s._('Translate')

    @p
    def fileOpenFailed(s):
        return s._('File Opening Failed.')

    @p
    def fileOpenFailedText(s):
        return s._('File could not be opened.')

