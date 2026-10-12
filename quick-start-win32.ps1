#!/usr/bin/env pwsh

# Windows check
if ($PSVersionTable.PSVersion.Major -gt 5)
{
    if (-Not $IsWindows)
    {
        Write-Output "This script is only for windows."
        Break
    }
}

# check for vcpkg
$vcpkgDefaultLoc = "$env:USERPROFILE/.vcpkg"
if (-Not $env:VCPKG_ROOT)
{
    Write-Host "VCPKG_ROOT Not found, checking for $vcpkgDefaultLoc"
    if (-Not (Test-Path $vcpkgDefaultLoc/vcpkg.exe))
    {
        Write-Host "vcpkg not found; will clone vcpkg"
        Remove-Item -Force -Recurse $vcpkgDefaultLoc
        git clone https://github.com/microsoft/vcpkg.git "$vcpkgDefaultLoc"

        Push-Location "$vcpkgDefaultLoc"
        & ./bootstrap-vcpkg.bat
        Pop-Location
    }
    else
    {
        Write-Host "vcpkg.exe found, will pull to latest"
        Push-Location "$vcpkgDefaultLoc"
        git pull --autostash
        Pop-Location
    }

    $env:VCPKG_ROOT = $vcpkgDefaultLoc
}
else
{
    Write-Host "Using VCPKG_ROOT at $($env:VCPKG_ROOT)"
}

# Launch-VsDevShell.ps1 below overwrites VCPKG_ROOT with Visual Studio's own bundled
# vcpkg, a stripped snapshot with no .git and no bootstrap-vcpkg.bat. The msvc/release
# preset takes its toolchain from $env:VCPKG_ROOT (see base/vcpkg in
# cmake-preset-files/base-presets.json), so without restoring it afterwards the configure
# below would silently use that snapshot instead of the clone resolved above.
$vcpkgRoot = $env:VCPKG_ROOT

# check for visual studio
$vsInfo = Get-CimInstance MSFT_VSInstance -Namespace root/cimv2/vs
if ($vsInfo -eq $null)
{
    Write-Output "Visual Studio not found. Please install visual studio."
    Break
}

Write-Output "Using $($vsInfo.Name) at $($vsInfo.InstallLocation)"
& "$($vsInfo.InstallLocation)\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation

$env:VCPKG_ROOT = $vcpkgRoot

cmake --preset msvc/release

Write-Output "Configuration done. Please run
cmake --build --preset msvc/release --target asy-with-basefiles
to build Asymptote."

Exit-PSSession
