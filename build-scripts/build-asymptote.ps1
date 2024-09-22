#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Script to build asymptote
.DESCRIPTION
    Builds asymptote installer file
.PARAMETER Version
    Specifies Asymptote version to build. If not given, will automatically determine version from configure.ac.
#>
param(
    [AllowEmptyString()]
    [Parameter()]
    [string]$Version
)
$usageString="Usage: $PSCommandPath -Version <version>"

# ----------------------------------------------------
# checking documentation files
$extfilesRoot="Z:\asy\asydoc"
$requiredDocumentationFiles=@(
    "asymptote.sty"
    "asymptote.pdf"
    "asy-latex.pdf"
    "CAD.pdf"
    "TeXShopAndAsymptote.pdf"
    "asyRefCard.pdf"
    "latexusage.pdf"
)

$hasDocFiles=$true
foreach ($requiredDocFile in $requiredDocumentationFiles) {
    if (-Not (Test-Path -PathType leaf "$extfilesRoot/$requiredDocFile")) {
        $hasDocFiles=$false
        Write-Error "$requiredDocFile not found.
Please ensure $requiredDocFile is available in $extfilesRoot directory"
    }
}

if (-Not $hasDocFiles) {
    Write-Error "Documentation file(s) are not present in $extFilesRoot directory. Will not build asymptote."
    Break
}

# ----------------------------------------------------
# tools cache
$toolscacheRoot="tools-cache"
New-Item -ItemType Directory -Path $toolscacheRoot -Force

$useToolsCacheVcpkg=$false

# tools cache variables
$vcpkgSha256="e590c2b30c08caf1dd8d612ec602a003f9784b7d"

# vcpkg
if (-Not $env:VCPKG_ROOT)
{
    $vcpkgToolsCacheLoc = "$toolscacheRoot/vcpkg"
    Write-Host "VCPKG_ROOT Not found, checking for $vcpkgToolsCacheLoc"
    if (-Not (Test-Path -PathType Container $vcpkgToolsCacheLoc))
    {
        git clone https://github.com/microsoft/vcpkg.git "$vcpkgToolsCacheLoc"
    }
    else
    {
        Write-Host "vcpkg directory found"
    }
    git -C $vcpkgToolsCacheLoc fetch
    git -C $vcpkgToolsCacheLoc reset --hard $vcpkgSha256

    if (-Not (Test-Path $vcpkgToolsCacheLoc/vcpkg.exe))
    {
        Push-Location $vcpkgToolsCacheLoc
        & ./bootstrap-vcpkg.bat
        Pop-Location
    }

    $useToolsCacheVcpkg=true
}
else
{
    Write-Host "Using VCPKG_ROOT at $($env:VCPKG_ROOT)"
}

# ------------------------------------------------------
# checking for NSIS
$makeNsisLoc=$null

# checking for downloaded nsis
if ($null -eq $makeNsisLoc)
{
    $nsisToolsCacheRoot="$toolscacheRoot/nsis"
    $downloadedNsis = "$nsisToolsCacheRoot/makensis.exe"
    if (Test-Path -PathType leaf $downloadedNsis)
    {
        Write-Host "Found downloaded NSIS at $downloadedNsis"
        $makeNsisLoc=$downloadedNsis
    }
}

# checking registry & install location
if ($null -eq $makeNsisLoc)
{
    $nsisInstallEntry = Get-ItemProperty `
    -Path HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\NSIS `
    -Name InstallLocation `
    -ErrorAction SilentlyContinue

    if ($null -ne $nsisInstallEntry)
    {
        # entry found for registry
        $nsisTestPath = "$( $nsisInstallEntry.InstallLocation )/makensis.exe"
        if (Test-Path -PathType leaf $nsisTestPath)
        {
            Write-Host "Found installed NSIS. Using NSIS at $nsisTestPath"
            $makeNsisLoc = $nsisTestPath
        }
    }
}

# check in tools-cache/nsis
if ($null -Eq $makeNsisLoc)
{
    $nsisPathExec = Get-Command makensis -ErrorAction ignore
    if ($null -Ne $nsisPathExec)
    {
        Write-Host "Found NSIS in PATH at $($makeNsisLoc.Path)".
        $makeNsisLoc=$nsisPathExec.Path
    }
}

# nsis not found
if ($null -eq $makeNsisLoc)
{
    Write-Error "Cannot find NSIS. Please either
(1, Recommended) Install NSIS from https://sourceforge.net/projects/nsis/
(2) Download NSIS ZIP and extract the files to $toolscacheRoot so that $nsisToolsCacheRoot/makensis.exe
    is available
"
    Break
}

# ------------------------------------------------
# python
$pyVenvLocation="$toolscacheRoot/pyxasy"
$pyXasyActivateScript="$pyVenvLocation/Scripts/activate.ps1"
if (-Not (Test-Path -PathType leaf $pyXasyActivateScript))
{
    python -m virtualenv $pyVenvLocation
}

# ----------------------------------------------------
# cloning asymptote
if (Test-Path asymptote)
{
    Remove-Item -Force -Recurse asymptote
}
# TODO: Once this is merged into master, the "-b msvc-suppoort-make" argument which
#       tells git to checkout to msvc-support-make branch can be removed
git clone --depth=1 -b msvc-support-make https://github.com/vectorgraphics/asymptote
Copy-Item -Recurse -Force -Path "$extfilesRoot" -Destination "asymptote/extfiles"

# ----------------------------------------------------
# determine version, if not given in arguments

if (0 -eq $Version.Length) {
    Write-Host "Version not given; will try to determine version"
    Push-Location asymptote
    $Version=python generate_asy_ver_info.py --version-for-release
    if (0 -ne $LASTEXITCODE) {
        Write-Error $usageString
        Write-Error "Cannot automatically determine release asymptote version. Please specify the version manually"
        Pop-Location
        Break
    }
    Pop-Location
    Write-Host "Asymptote version is $Version"
}

# ----------------------------------------------------
# build GUI
& $pyXasyActivateScript
Push-Location asymptote/GUI
& python -m pip install -r requirements.txt -r requirements.dev.txt
& python buildtool.py build --version-override=$Version
Pop-Location


# ----------------------------------------------------

function buildAsy($preset, $cfgDir) {
    # build C++ side
    Import-VisualStudioVars -Architecture x64
    Push-EnvironmentBlock
    $env:ASY_VERSION_OVERRIDE = $Version
    if ($useToolsCacheVcpkg)
    {
        $env:VCPKG_ROOT = $vcpkgToolsCacheLoc
    }
    Push-Location asymptote
    cmake --preset $preset
    Pop-Location
    cmake --build $cfgDir --target asy-pre-nsis-targets -j
    Pop-EnvironmentBlock  # ASY_VERSION_OVERRIDE, VCPKG_ROOT
    Pop-EnvironmentBlock  # Visual studio vars
    # install to pre-installation root
}

buildAsy msvc/release-with-external-doc-files asymptote/cmake-build-msvc/release
cmake --install asymptote/cmake-build-msvc/release --component asy-pre-nsis

# ------------------------------------------------------
# Generate NSIS installer file
& ./asymptote/cmake-install-w32-nsis-release/build-asy-installer.ps1 "$makeNsisLoc"


$asySetupFile="./asymptote/cmake-install-w32-nsis-release/asymptote-$Version-setup.exe"

if (Test-Path -PathType leaf "asymptote-$Version-setup.exe")
{
    Write-Host "Found old setup file. Will delete the file."
    Remove-Item -Force "asymptote-$Version-setup.exe"
}

if (Test-Path -PathType leaf $asySetupFile)
{
    Move-Item $asySetupFile . -Force
}
else
{
    Write-Error "Asymptote setup file not built."
    Break
}

# ------------------------------------------------------
# building for CTAN

buildAsy msvc/release-with-external-doc-file-ctan asymptote/cmake-build-msvc/release
New-Item -ItemType Directory -Path CTAN -Force
Copy-Item asymptote/cmake-build-msvc/release/asy.exe -Destination CTAN/asy.exe
