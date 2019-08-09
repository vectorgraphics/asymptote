#!/usr/bin/env python3
###########################################################################
#
# xasyOptions provides a mechanism for storing and restoring a user's
# preferences.
#
#
# Author: Orest Shardt
# Created: June 29, 2007
#
###########################################################################

import json
import sys
import io
import os
import platform
import shutil

import configs

try:
    import cson
except ModuleNotFoundError:
    cson = None

try:
    pass
#     import yaml
except ModuleNotFoundError:
    yaml = None

class xasyOptions:
    def defaultOptions(self):
        if self._defaultOptions is None:
            f = io.open(self._defaultOptLocation)
            try:
                opt = cson.loads(f.read())
            finally:
                f.close()
            self._defaultOptions = opt
        return self._defaultOptions

    def overrideSettings(self):
        settingsName = platform.system()

        if settingsName not in self.options:
            return

        for key in self.options[settingsName]:
            self.options[key] = self.options[settingsName][key]
    
    
    def settingsFileLocation(self):
        folder = os.path.expanduser("~/.asy/")

        searchOrder = ['.cson', '.yaml', '.json', '']

        searchIndex = 0
        found = False
        currentFile = ''
        while searchIndex < len(searchOrder) and not found:
            currentFile = os.path.join(folder, self.configName + searchOrder[searchIndex])
            if os.path.isfile(currentFile):
                found = True
            searchIndex += 1
        
        if found:
            return os.path.normcase(currentFile)
        else:
            return os.path.normcase(os.path.join(folder, self.configName + '.cson'))

    def __init__(self, configName, defaultConfigLocation):
        self.configName = configName
        self.defaultConfigName = defaultConfigLocation

        self._defaultOptions = None
        self._defaultOptLocation = os.path.join(defaultConfigLocation)

        self.options = self.defaultOptions()
        self.load()
        self.overrideSettings()

    def __getitem__(self, item):
        return self.options[item]

    def __setitem__(self, key, value):
        self.options[key] = value

    def load(self):
        fileName = self.settingsFileLocation()
        if not os.path.exists(fileName):
            # make folder
            thedir = os.path.dirname(fileName)
            if not os.path.exists(thedir):
                os.makedirs(thedir)
            if not os.path.isdir(thedir):
                raise Exception("Configuration folder path does not point to a folder")
            self.setDefaults()
        f = io.open(fileName, 'r')
        try:
            ext = os.path.splitext(fileName)[1]
            if ext == '.cson':
                if cson is None:
                    raise ModuleNotFoundError
                newOptions = cson.loads(f.read())
            elif ext in {'.yml', '.yaml'}:
                if yaml is None:
                    raise ModuleNotFoundError
                newOptions = yaml.load(f)
            else:
                newOptions = json.loads(f.read())
        except (IOError, ModuleNotFoundError):
            self.setDefaults()
        else:
            for key in self.options.keys():
                if key in newOptions:
                    assert isinstance(newOptions[key], type(self.options[key]))
                else:
                    newOptions[key] = self.options[key]
            self.options = newOptions
        finally:
            f.close()

    def setDefaults(self):
        self.options = self.defaultOptions()
        if sys.platform[:3] == 'win':  # for windows, wince, win32, etc
            # setAsyPathFromWindowsRegistry()
            pass
        folder = os.path.expanduser("~/.asy/")
        defaultPath = os.path.join(folder, self.configName + '.cson')
        shutil.copy2(self._defaultOptLocation, defaultPath)
        

# TODO: Figure out how to merge this back.
"""
def setAsyPathFromWindowsRegistry():
    if os.name == 'nt':
        import _winreg as registry
        # test both registry locations
        try:
            key = registry.OpenKey(registry.HKEY_LOCAL_MACHINE,
                                   "Software\\Microsoft\\Windows\\CurrentVersion\\App Paths\\Asymptote")
            options['asyPath'] = registry.QueryValueEx(key, "Path")[0] + "\\asy.exe"
            registry.CloseKey(key)
        except:
            key = registry.OpenKey(registry.HKEY_LOCAL_MACHINE,
                                   "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Asymptote")
            options['asyPath'] = registry.QueryValueEx(key, "InstallLocation")[0] + "\\asy.exe"
            registry.CloseKey(key)
"""


class BasicConfigs:
    _configPath = list(configs.__path__)[0]
    defaultOpt = xasyOptions(
        'xasyconfig', os.path.join(_configPath, 'xasyconfig.cson'))
    keymaps = xasyOptions('xasykeymap', os.path.join(
        _configPath, 'xasykeymap.cson'))
