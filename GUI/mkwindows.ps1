$uipath = 'windows/'
$outputpath = 'pyUIClass/'
$files = Get-ChildItem -path $uipath -Filter *.ui -file

foreach ($file in $files)
{
    $pyuifile = pyuic5 ($uipath + $file);
    set-content -path ($outputpath + '//' +  $file.basename + '.py') -value $pyuifile;
}
write-host Done!;