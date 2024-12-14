#!/usr/bin/env pwsh
$lintable_files=$(python "$PSScriptRoot/print_non_gui_py_files_for_linting.py")

isort --settings-file="$PSScriptRoot/../.isort.cfg" --check $lintable_files
pylint --rcfile="$PSScriptRoot/../.pylintrc" $lintable_files
