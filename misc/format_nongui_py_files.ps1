#!/usr/bin/env pwsh
$lintable_files=$(python "$PSScriptRoot/print_non_gui_py_files_for_linting.py")
isort --settings-file="$PSScriptRoot/../.isort.cfg" $lintable_files
black $lintable_files
