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

import sys
import os
import platform
import shutil
import configs
import cson

from pathlib import Path
from typing import Any

try:
    from platformdirs import user_config_dir, user_state_dir
except ModuleNotFoundError:
    def user_config_dir(appname: str, *args, **kwargs):
        path = os.environ.get("XDG_CONFIG_HOME", "")
        if not path.strip():
            path = os.path.expanduser("~/.local/share")
        return os.path.join(path, appname)

    def user_state_dir(appname: str, *args, **kwargs):
        path = os.environ.get("XDG_STATE_HOME", "")
        if not path.strip():
            path = os.path.expanduser("~/.local/state")
        return os.path.join(path, appname)


class xasyOptions:
    def __init__(self, configName: str, defaultConfigLocation: Path):
        self.configName = configName
        self._defaultOptLocation = defaultConfigLocation

        self.options: "dict[str, Any]" = {}
        self.load()

    def defaultOptions(self):
        opt: "dict[str, Any]" = cson.loads(self._defaultOptLocation.read_text())  # type: ignore
        return opt

    def overrideSettings(self):
        """Apply OS specific overrides"""
        settingsName = platform.system()

        if settingsName not in self.options:
            return

        for key in self.options[settingsName]:
            self.options[key] = self.options[settingsName][key]

    def settingsFileLocation(self):
        folders = [Path(user_config_dir("asymptote")), Path.home() / ".asy"]
        searchOrder = [".cson", ""]

        for folder in folders:
            for ext in searchOrder:
                path = folder / f"{self.configName}{ext}"
                if path.is_file():
                    return str(path)

        folders[0].mkdir(exist_ok=True, parents=True)
        return str(folders[0] / f"{self.configName}.cson")

    def __getitem__(self, key):
        return self.options[key]

    def __contains__(self, key):
        return key in self.options

    def get(self, key, default=None):
        if key not in self.options:
            return default
        return self.options[key]

    def __setitem__(self, key, value):
        self.options[key] = value

    def load(self):
        """
        Loads settings/keymaps to the `self.options` attribute.

        This method follows the order of precedence:

        1. Loads the default options.
        2. Applies the current options.
        3. Applies options from the configuration file.
        4. Applies OS-specific overrides to adjust options based on the current platform.
        """
        fileName = Path(self.settingsFileLocation())

        defaults = self.defaultOptions()

        newOptions = {}
        if fileName.exists():
            newOptions: "dict[str, Any]" = cson.loads(fileName.read_text())  # type: ignore

        # assert types match
        for source in [defaults, self.options]:
            common_keys = set(source.keys()).intersection(newOptions.keys())
            for key in common_keys:
                old_value = source[key]
                if old_value is not None:
                    assert isinstance(newOptions[key], type(old_value))

        self.options = {**defaults, **self.options, **newOptions}
        self.overrideSettings()

    def setDefaults(self):
        """Reset options/keymaps."""
        self.options = self.defaultOptions()
        if sys.platform[:3] == "win":  # for windows, wince, win32, etc
            # setAsyPathFromWindowsRegistry()
            pass

        shutil.copy2(self._defaultOptLocation, self.settingsFileLocation())


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


class xasyOpenRecent:
    def settingsFileLocation(self, ensure_exists: bool = True):
        folders = [Path(user_state_dir("asymptote")), Path.home() / ".asy"]
        file = "xasyrecents.txt"

        for folder in folders:
            path = folder / file
            if path.exists():
                return path

        path = folders[0] / file
        if ensure_exists:
            path.parent.mkdir(exist_ok=True, parents=True)
            path.touch()
        return path

    def insert(self, path: str):
        """
        Adds a file path to the top of the recent files list,
        or moves it to the top if it already exists in the list.
        """
        path = path.strip()
        fileName = self.settingsFileLocation()

        lines = [line.strip() for line in fileName.read_text().splitlines()]

        paths = [p for p in lines if p != path]
        paths.insert(0, path)

        fileName.write_text("\n".join(paths) + "\n")

    @property
    def pathList(self):
        """Retrieves the current list of recently opened file paths.

        Loads the file paths from the storage file and ensures
        that any paths that no longer exist (i.e., are missing or deleted)
        are removed from the list.
        """

        fileName = self.settingsFileLocation()
        lines = [line.strip() for line in fileName.read_text().splitlines()]

        existing = [path for path in lines if Path(path).resolve().is_file()]
        if existing != lines:
            fileName.write_text("\n".join(existing) + "\n")

        return existing

    def clear(self):
        """Remove all items from the storage file."""
        self.settingsFileLocation().write_text("")


class BasicConfigs:
    _configPath = Path(list(configs.__path__)[0])
    defaultOpt = xasyOptions("xasyconfig", _configPath / "xasyconfig.cson")
    keymaps = xasyOptions("xasykeymap", _configPath / "xasykeymap.cson")
    openRecent = xasyOpenRecent()
