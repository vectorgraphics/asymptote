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

!define Environ 'HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"'

; AddToPath - Appends dir to PATH
;   (does not work on Win9x/ME)
;
; Usage:
;   Push "dir"
;   Call AddToPath

Function AddToPath
  Exch $0
  Push $1
  Push $2
  Push $3
  Push $4

  ; NSIS ReadRegStr returns empty string on string overflow
  ; Native calls are used here to check actual length of PATH

  ; $4 = RegOpenKey(HKEY_CURRENT_USER, "Environment", &$3)
  System::Call "advapi32::RegOpenKey(i 0x80000001, t'Environment', *i.r3) i.r4"
  IntCmp $4 0 0 done done
  ; $4 = RegQueryValueEx($3, "PATH", (DWORD*)0, (DWORD*)0, &$1, ($2=NSIS_MAX_STRLEN, &$2))
  ; RegCloseKey($3)
  System::Call "advapi32::RegQueryValueEx(i $3, t'PATH', i 0, i 0, t.r1, *i ${NSIS_MAX_STRLEN} r2) i.r4"
  System::Call "advapi32::RegCloseKey(i $3)"

  ${If} $4 = 234 ; ERROR_MORE_DATA
    DetailPrint "AddToPath: original length $2 > ${NSIS_MAX_STRLEN}"
    MessageBox MB_OK "PATH not updated, original length $2 > ${NSIS_MAX_STRLEN}" /SD IDOK
    Goto done
  ${EndIf}

  ${If} $4 <> 0 ; NO_ERROR
    ${If} $4 <> 2 ; ERROR_FILE_NOT_FOUND
      DetailPrint "AddToPath: unexpected error code $4"
      Goto done
    ${EndIf}
    StrCpy $1 ""
  ${EndIf}

  ; Check if already in PATH
  Push "$1;"
  Push "$0;"
  Call StrStr
  Pop $2
  StrCmp $2 "" 0 done
  Push "$1;"
  Push "$0\;"
  Call StrStr
  Pop $2
  StrCmp $2 "" 0 done

  ; Prevent NSIS string overflow
  StrLen $2 $0
  StrLen $3 $1
  IntOp $2 $2 + $3
  IntOp $2 $2 + 2 ; $2 = strlen(dir) + strlen(PATH) + sizeof(";")
  ${If} $2 > ${NSIS_MAX_STRLEN}
    DetailPrint "AddToPath: new length $2 > ${NSIS_MAX_STRLEN}"
    MessageBox MB_OK "PATH not updated, new length $2 > ${NSIS_MAX_STRLEN}." /SD IDOK
    Goto done
  ${EndIf}

  ; Append dir to PATH
  DetailPrint "Add to PATH: $0"
  StrCpy $2 $1 1 -1
  ${If} $2 == ";"
    StrCpy $1 $1 -1 ; remove trailing ';'
  ${EndIf}
  ${If} $1 != "" ; no leading ';'
    StrCpy $0 "$1;$0"
  ${EndIf}
  WriteRegExpandStr ${Environ} "PATH" $0
  SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000

done:
  Pop $4
  Pop $3
  Pop $2
  Pop $1
  Pop $0
FunctionEnd

; StrStr - find substring in a string
;
; Usage:
;   Push "this is some string"
;   Push "some"
;   Call StrStr
;   Pop $0 ; "some string"

!macro StrStr un
Function ${un}StrStr
  Exch $R1 ; $R1=substring, stack=[old$R1,string,...]
  Exch     ;                stack=[string,old$R1,...]
  Exch $R2 ; $R2=string,    stack=[old$R2,old$R1,...]
  Push $R3
  Push $R4
  Push $R5
  StrLen $R3 $R1
  StrCpy $R4 0
  ; $R1=substring, $R2=string, $R3=strlen(substring)
  ; $R4=count, $R5=tmp
  ${Do}
    StrCpy $R5 $R2 $R3 $R4
    ${IfThen} $R5 == $R1 ${|} ${ExitDo} ${|}
    ${IfThen} $R5 == ""  ${|} ${ExitDo} ${|}
    IntOp $R4 $R4 + 1
  ${Loop}
  StrCpy $R1 $R2 "" $R4
  Pop $R5
  Pop $R4
  Pop $R3
  Pop $R2
  Exch $R1 ; $R1=old$R1, stack=[result,...]
FunctionEnd
!macroend
!insertmacro StrStr ""
!insertmacro StrStr "un."

Function un.RemoveFromPath
  Exch $0
  Push $1
  Push $2
  Push $3
  Push $4
  Push $5
  Push $6

  ReadRegStr $1 ${Environ} "PATH"
  StrCpy $5 $1 1 -1
  ${If} $5 != ";"
    StrCpy $1 "$1;" ; ensure trailing ';'
  ${EndIf}
  Push $1
  Push "$0;"
  Call un.StrStr
  Pop $2 ; pos of our dir
  StrCmp $2 "" done

  DetailPrint "Remove from PATH: $0"
  StrLen $3 "$0;"
  StrLen $4 $2
  StrCpy $5 $1 -$4 ; $5 is now the part before the path to remove
  StrCpy $6 $2 "" $3 ; $6 is now the part after the path to remove
  StrCpy $3 "$5$6"
  StrCpy $5 $3 1 -1
  ${If} $5 == ";"
    StrCpy $3 $3 -1 ; remove trailing ';'
  ${EndIf}
  WriteRegExpandStr ${Environ} "PATH" $3
  SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000

done:
  Pop $6
  Pop $5
  Pop $4
  Pop $3
  Pop $2
  Pop $1
  Pop $0
FunctionEnd

Section "Asymptote" SEC01
  SetOutPath "$INSTDIR"
  Push "$INSTDIR"
  Call AddToPath

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
  CreateShortCut "$DESKTOP\Xasy.lnk" "$INSTDIR\xasy.py"
  CreateShortCut "$SMPROGRAMS\$ICONS_GROUP\Xasy.lnk" "$INSTDIR\xasy.py"
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

  ; Remove install dir from PATH
  Push "$INSTDIR"
  Call un.RemoveFromPath

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
