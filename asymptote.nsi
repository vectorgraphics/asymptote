!define PRODUCT_NAME "Asymptote"
!include AsymptoteInstallInfo.nsi
!define PRODUCT_WEB_SITE "http://asymptote.sourceforge.net/"
!define PRODUCT_DIR_REGKEY "Software\Microsoft\Windows\CurrentVersion\App Paths\Asymptote"
!define PRODUCT_FILE_TYPE_REGKEY1 "Software\Classes\.asy"
!define PRODUCT_FILE_TYPE_REGKEY2 "Software\Classes\ASYFile\shell\open\command"
!define PRODUCT_UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"
!define PRODUCT_UNINST_ROOT_KEY "HKLM"
!define PRODUCT_STARTMENU_REGVAL "NSIS:StartMenuDir"

SetCompressor lzma
XPStyle On

; MUI 1.67 compatible ------
!include "MUI.nsh"
!include "LogicLib.nsh"
!include "lnkX64IconFix.nsh"

; MUI Settings
!define MUI_ABORTWARNING
!define MUI_ICON "asy.ico"
!define MUI_UNICON "asy.ico"

; Welcome page
!insertmacro MUI_PAGE_WELCOME
; License page
!insertmacro MUI_PAGE_LICENSE "LICENSE"
;Components page
; don't bother with this until there are other components to install
; e.g.: possibility to automatically detect presence of, download, and install python, miktex, ImageMagick, etc
;!insertmacro MUI_PAGE_COMPONENTS
; Directory page
!insertmacro MUI_PAGE_DIRECTORY
; Start menu page
var ICONS_GROUP
!define MUI_STARTMENUPAGE_NODISABLE
!define MUI_STARTMENUPAGE_DEFAULTFOLDER "Asymptote"
!define MUI_STARTMENUPAGE_REGISTRY_ROOT "${PRODUCT_UNINST_ROOT_KEY}"
!define MUI_STARTMENUPAGE_REGISTRY_KEY "${PRODUCT_UNINST_KEY}"
!define MUI_STARTMENUPAGE_REGISTRY_VALUENAME "${PRODUCT_STARTMENU_REGVAL}"
!insertmacro MUI_PAGE_STARTMENU Application $ICONS_GROUP
; Instfiles page
!insertmacro MUI_PAGE_INSTFILES
; Finish page
;!define MUI_FINISHPAGE_RUN "$INSTDIR\asy.bat"
;!define MUI_FINISHPAGE_SHOWREADME "$INSTDIR\asymptote.pdf"
!define MUI_FINISHPAGE_LINK ${PRODUCT_WEB_SITE}
!define MUI_FINISHPAGE_LINK_LOCATION ${PRODUCT_WEB_SITE}
!insertmacro MUI_PAGE_FINISH

; Uninstaller pages
!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

; Language files
!insertmacro MUI_LANGUAGE "English"

; Reserve files
!insertmacro MUI_RESERVEFILE_INSTALLOPTIONS

; MUI end ------

Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile "asymptote-${PRODUCT_VERSION}-setup.exe"
InstallDir "$PROGRAMFILES64\Asymptote"
InstallDirRegKey HKLM "${PRODUCT_DIR_REGKEY}" ""
ShowInstDetails show
ShowUnInstDetails show

Section "Asymptote" SEC01
  SetOutPath "$INSTDIR"
  SetOverwrite try
  File /r build-${PRODUCT_VERSION}\*

  FileOpen $0 $INSTDIR\asy.bat w

  FileWrite $0 "@ECHO OFF"
  FileWriteByte $0 "13" 
  FileWriteByte $0 "10" 

  FileWrite $0 "set CYGWIN=nodosfilewarning"
  FileWriteByte $0 "13" 
  FileWriteByte $0 "10" 

  FileWrite $0 '"$INSTDIR\asy.exe" %*'
  FileWriteByte $0 "13" 
  FileWriteByte $0 "10" 

  FileWrite $0 "if %errorlevel% == 0 exit /b"
  FileWriteByte $0 "13" 
  FileWriteByte $0 "10" 

  FileWrite $0 "echo."
  FileWriteByte $0 "13" 
  FileWriteByte $0 "10" 

  FileWrite $0 "PAUSE"
  FileWriteByte $0 "13" 
  FileWriteByte $0 "10" 

  FileClose $0

; Shortcuts
  !insertmacro MUI_STARTMENU_WRITE_BEGIN Application
  CreateDirectory "$SMPROGRAMS\$ICONS_GROUP"
  SetOutPath "%USERPROFILE%"
  CreateShortCut "$SMPROGRAMS\$ICONS_GROUP\Asymptote.lnk" "$INSTDIR\asy.bat" "" "$INSTDIR\asy.ico"
  ${lnkX64IconFix} "$SMPROGRAMS\$ICONS_GROUP\Asymptote.lnk"
  CreateShortCut "$DESKTOP\Asymptote.lnk" "$INSTDIR\asy.bat" "" "$INSTDIR\asy.ico"
  ${lnkX64IconFix} "$DESKTOP\Asymptote.lnk"
  CreateShortCut "$DESKTOP\Xasy.lnk" "$INSTDIR\GUI\xasy.py"
  CreateShortCut "$SMPROGRAMS\$ICONS_GROUP\Xasy.lnk" "$INSTDIR\GUI\xasy.py"
  SetOutPath "$INSTDIR"
  !insertmacro MUI_STARTMENU_WRITE_END
SectionEnd

Section "Tester"

SectionEnd

Section -AdditionalIcons
  !insertmacro MUI_STARTMENU_WRITE_BEGIN Application
  WriteIniStr "$INSTDIR\${PRODUCT_NAME}.url" "InternetShortcut" "URL" "${PRODUCT_WEB_SITE}"
  CreateShortCut "$SMPROGRAMS\$ICONS_GROUP\Website.lnk" "$INSTDIR\${PRODUCT_NAME}.url"
  CreateShortCut "$SMPROGRAMS\$ICONS_GROUP\Uninstall.lnk" "$INSTDIR\uninst.exe"
  !insertmacro MUI_STARTMENU_WRITE_END
SectionEnd

Section -Post
  WriteUninstaller "$INSTDIR\uninst.exe"
  ;create registry keys with information needed to run asymptote
  WriteRegStr HKLM "${PRODUCT_DIR_REGKEY}" "" "$INSTDIR\asy.exe"
  WriteRegStr HKLM "${PRODUCT_DIR_REGKEY}" "Path" "$INSTDIR"
  WriteRegStr HKLM "${PRODUCT_FILE_TYPE_REGKEY1}" "" "ASYFile"
  WriteRegStr HKLM "${PRODUCT_FILE_TYPE_REGKEY2}" "" '"$INSTDIR\asy.bat" "%1"'
  WriteRegDWORD HKLM "SOFTWARE\Cygwin" "heap_chunk_in_mb" 0xFFFFFF00
  ReadRegStr $0 HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "CYGWIN"
  ${If} $0 == ""
    WriteRegExpandStr HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "CYGWIN" "nodosfilewarning"
    SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000
  ${Endif}
  
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayName" "$(^Name)"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "UninstallString" "$INSTDIR\uninst.exe"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayIcon" "$INSTDIR\asy.exe"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayVersion" "${PRODUCT_VERSION}"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "URLInfoAbout" "${PRODUCT_WEB_SITE}"
SectionEnd


Function un.onUninstSuccess
  HideWindow
  MessageBox MB_ICONINFORMATION|MB_OK "$(^Name) was successfully removed from your computer."
FunctionEnd

Section Uninstall
  !insertmacro MUI_STARTMENU_GETFOLDER "Application" $ICONS_GROUP
  Delete "$INSTDIR\${PRODUCT_NAME}.url"
  Delete "$INSTDIR\uninst.exe"
  !include AsymptoteUninstallList.nsi
  Delete "$INSTDIR\asy.bat"
  RMDir "$INSTDIR"
  
  Delete "$SMPROGRAMS\$ICONS_GROUP\Uninstall.lnk"
  Delete "$SMPROGRAMS\$ICONS_GROUP\Website.lnk"
  Delete "$DESKTOP\Asymptote.lnk"
  Delete "$DESKTOP\Xasy.lnk"
  Delete "$SMPROGRAMS\$ICONS_GROUP\Asymptote.lnk"
  Delete "$SMPROGRAMS\$ICONS_GROUP\Xasy.lnk"
  RMDir "$SMPROGRAMS\$ICONS_GROUP"


  DeleteRegKey ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}"
  DeleteRegKey HKLM "${PRODUCT_DIR_REGKEY}"
  DeleteRegKey HKLM "${PRODUCT_FILE_TYPE_REGKEY1}"
  DeleteRegKey HKLM "${PRODUCT_FILE_TYPE_REGKEY2}"
  SetAutoClose true
SectionEnd
