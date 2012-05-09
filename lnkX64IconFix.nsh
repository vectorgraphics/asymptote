/******************************************************************************
    WORKAROUND - lnkX64IconFix
        This snippet was developed to address an issue with Windows 
        x64 incorrectly redirecting the shortcuts icon from $PROGRAMFILES32 
        to $PROGRAMFILES64.
 
    See Forum post: http://forums.winamp.com/newreply.php?do=postreply&t=327806
 
    Example:
        CreateShortcut "$SMPROGRAMS\My App\My App.lnk" "$INSTDIR\My App.exe" "" "$INSTDIR\My App.exe"
        ${lnkX64IconFix} "$SMPROGRAMS\My App\My App.lnk"
 
    Original Code by Anders [http://forums.winamp.com/member.php?u=70852]
 ******************************************************************************/
!ifndef ___lnkX64IconFix___
    !verbose push
    !verbose 0
 
    !include "LogicLib.nsh"
    !include "x64.nsh"
 
    !define ___lnkX64IconFix___
    !define lnkX64IconFix `!insertmacro _lnkX64IconFix`
    !macro _lnkX64IconFix _lnkPath
        !verbose push
        !verbose 0
        ${If} ${RunningX64}
            DetailPrint "WORKAROUND: 64bit OS Detected, Attempting to apply lnkX64IconFix"
            Push "${_lnkPath}"
            Call lnkX64IconFix
        ${EndIf}
        !verbose pop
    !macroend
 
    Function lnkX64IconFix ; _lnkPath
        Exch $5
        Push $0
        Push $1
        Push $2
        Push $3
        Push $4
        System::Call 'OLE32::CoCreateInstance(g "{00021401-0000-0000-c000-000000000046}",i 0,i 1,g "{000214ee-0000-0000-c000-000000000046}",*i.r1)i'
        ${If} $1 <> 0
            System::Call '$1->0(g "{0000010b-0000-0000-C000-000000000046}",*i.r2)'
            ${If} $2 <> 0
                System::Call '$2->5(w r5,i 2)i.r0'
                ${If} $0 = 0
                    System::Call '$1->0(g "{45e2b4ae-b1c3-11d0-b92f-00a0c90312e1}",*i.r3)i.r0'
                    ${If} $3 <> 0
                        System::Call '$3->5(i 0xA0000007)i.r0'
                        System::Call '$3->6(*i.r4)i.r0'
                        ${If} $0 = 0 
                            IntOp $4 $4 & 0xffffBFFF
                            System::Call '$3->7(ir4)i.r0'
                            ${If} $0 = 0 
                                System::Call '$2->6(i0,i0)'
                                DetailPrint "WORKAROUND: lnkX64IconFix Applied successfully"
                            ${EndIf}
                        ${EndIf}
                        System::Call $3->2()
                    ${EndIf}
                ${EndIf}
                System::Call $2->2()
            ${EndIf}
            System::Call $1->2()
        ${EndIf} 
        Pop $4
        Pop $3
        Pop $2
        Pop $1
        Pop $0
    FunctionEnd
    !verbose pop
!endif
