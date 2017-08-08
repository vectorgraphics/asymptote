$uipath = 'windows/'
$files = Get-ChildItem -path $uipath -Filter *.ui -file

foreach ($file in $files)
{
    $pyuifile = pyuic5 ($uipath + $file);
    set-content -path ($file.basename + '.py') -value $pyuifile;
}
write-host Done!;