#!/usr/bin/env powershell
$uipath = 'windows/'
$respath = 'res/'
$outputpath = 'pyUIClass/'
$files = Get-ChildItem -path $uipath -Filter *.ui -file
$resfile = Get-ChildItem -path $respath -Filter *.qrc -file

foreach ($file in $files)
{
    $pyuifile = pyuic5 ($uipath + $file);
    set-content -path ($outputpath + '//' +  $file.basename + '.py') -value $pyuifile;
}

foreach ($file in $resfile)
{
    $pyuifile = pyrcc5 ($respath + $file);
    set-content -path ( $file.basename + '_rc.py') -value $pyuifile;
}

write-host Done!;