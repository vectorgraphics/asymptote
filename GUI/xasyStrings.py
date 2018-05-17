import gettext


class xasyString:
    def __init__(self, lang=None):
        if lang is None:
            self._ = gettext.gettext
        else:
            lng = gettext.translation('base', localedir='GUI/locale', languages=[lang])
            lng.install()
            self._ = lng.gettext

    @property
    def rotate(self):
        return self._('Rotate')

