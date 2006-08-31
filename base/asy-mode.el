;;; asy-mode.el

;; Copyright (C) 2006
;; Author: Philippe IVALDI 20 August 2006
;; Last modification: John Bowman 25 August 2006 
;;
;; This program is free software ; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation ; either version 2 of the License, or
;; (at your option) any later version.
;;
;; This program is distributed in the hope that it will be useful, but
;; WITHOUT ANY WARRANTY ; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
;; General Public License for more details.
;;
;; You should have received a copy of the GNU General Public License
;; along with this program ; if not, write to the Free Software
;; Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

;; Emacs mode for editing Asymptote source code.
;;
;; INSTALLATION:
;; Place this file (asy-mode.el) in your Emacs load path.
;; Copy and uncomment the following 2 lines to your .emacs initialization file:

;; (autoload 'asy-mode "asy-mode" "Asymptote major mode." t)
;; (setq auto-mode-alist (cons (cons "\\.asy$" 'asy-mode) auto-mode-alist))

;; For full functionality the two-mode-mode package should also be installed 
;; (http://www.dedasys.com/freesoftware/files/two-mode-mode.el).
;; The package texmathp is optional.

(define-derived-mode asy-mode objc-mode "Asymptote"
  "Emacs mode for editing Asymptote source code.
For full functionality the 'two-mode-mode package should also be installed 
(http://www.dedasys.com/freesoftware/files/two-mode-mode.el).
The package 'texmathp is optional.

This package provides two modes:
1- asy-mode:
  All the files with the extension '.asy' are edited in this mode, which provides the following features:
    * Syntax color highlighting;
    * Compiling and viewing current buffer with the key binding C-c C-c;
    * Moving cursor to the error by pressing the key F4.
    * Showing the available function prototypes for the command at the cursor with the key binding C-c ?
    * Inserting template by pressing the key F3.
       - For example ife<F3> gives:
          if (*)
            {
              **
            }
          else
            {
              ***
            }

       The cursor is *. Press M-Right to go to **, M-Right again for ***
       - Look at the code after the comment 'Templates appended to asy-tempo-tags' to add your entries.
    * Compiling and viewing a TeX document linked with the current buffer (usually a document that includes the output picture).
       To link a Tex document try 'M-x asy-set-master-tex' follow by C-Return (see descriptions further of the key binding C-Return, C-S-Return, M-Return, M-S-Return etc within 2- lasy-mode)

2- lasy-mode
 Editing a TeX file that contains Asymptote code is facilitated with the hybrid mode 'lasy-mode.
 Toggle lasy-mode with M-x lasy-mode.
 In this hybrid mode the major mode is LaTeX when the cursor is in LaTeX code and becomes asy-mode when the cursor is between '\begin{asy}' and '\end{asy}'.
 All the features of asy-mode are provided and the key binding C-c C-c of asy-mode compiles and views only the code of the picture where is the cursor.
 Note that some keys binding are added to the LaTeX-mode-map in lasy-mode and also, if you want, in pure LaTeX-mode (customize the variable 'lasy-keep-key to accept or refuse the modifications in pure LaTeX-mode).
   * C-return : compile (if the buffer/file is modified) and view the postscript output with sequence [latex->[asy->latex]->dvips]->PSviewer
   * M-return : same with pdf output and with the sequence [pdflatex -shell-escape->[asy->pdflatex -shell-escape]]->PDFviewer
   * C-M-return : same with pdf output and with the sequence [latex->[asy->latex]->dvips->ps2pdf]->PSviewer
   * Add the Shift key to the sequence of keys to compile even if the file is not modified.

You can access this help within Emacs by the key binding C-h f asy-mode <RET>

BUGS:
This package has been created and tested with:
  - Linux Debian Sarge
  - GNU Emacs 22.0.50.1
  - AUCTeX 11.55
  - Asymptote 1.11
Report bugs to http://asymptote.sourceforge.net

Some variables can be customized: M-x customize-group <RET> asymptote <RET>."

  (setq c++-font-lock-extra-types (cons "real" c++-font-lock-extra-types))
  (setq c++-font-lock-extra-types (cons "pair" c++-font-lock-extra-types))
  (setq c++-font-lock-extra-types (cons "triple" c++-font-lock-extra-types))
  (setq c++-font-lock-extra-types (cons "transform" c++-font-lock-extra-types))
  (setq c++-font-lock-extra-types (cons "guide" c++-font-lock-extra-types))
  (setq c++-font-lock-extra-types (cons "path" c++-font-lock-extra-types))
  (setq c++-font-lock-extra-types (cons "pen" c++-font-lock-extra-types))
  (setq c++-font-lock-extra-types (cons "picture" c++-font-lock-extra-types))
  (setq c++-font-lock-extra-types (cons "frame" c++-font-lock-extra-types))
  (setq c++-font-lock-extra-types (cons "import" c++-font-lock-extra-types))
  (setq c++-font-lock-extra-types (cons "access" c++-font-lock-extra-types))
  (setq c++-font-lock-extra-types (cons "unravel" c++-font-lock-extra-types))
  (setq c++-font-lock-extra-types (cons "from" c++-font-lock-extra-types))
  (setq c++-font-lock-extra-types (cons "include" c++-font-lock-extra-types))
  
  (setq skeleton-pair t)
;; Next 3 lines disabled: end delimiters are treated as begin delimiters! (JCB)
;;  (define-key asy-mode-map "\"" 'skeleton-pair-insert-maybe)
;;  (define-key asy-mode-map "\'" 'skeleton-pair-insert-maybe)
;;  (define-key asy-mode-map "\$" 'skeleton-pair-insert-maybe)
;; Next 3 lines also disabled as they can make typing awkward. (JCB)
;;  (define-key asy-mode-map "\{" 'skeleton-pair-insert-maybe)
;;  (define-key asy-mode-map "\(" 'skeleton-pair-insert-maybe)
;;  (define-key asy-mode-map "\[" 'skeleton-pair-insert-maybe)
)

(defcustom asy-command-location ""
  "*If not in the path, you can put here the name of the directory containing Asy's binary files.
this variable must end in / (UNIX) or \ (MSWindows)."
  :type 'directory
  :group 'asymptote)

(defcustom asy-temp-dir temporary-file-directory
  "*The name of a directory for Asy's temporary files.
Such files are generated by functions like
`asy-compile' when lasy-mode is enable."
  :type 'directory
  :group 'asymptote)

(defcustom ps-view-command (cond
                            ((eq window-system 'x)
                             "gv")
                            ((eq window-system 'w32)
                             "gsview")
                            (t "_indefinite_ps_viewer_"))
  "Command to view a Postscript file."
  :type 'file
  :group 'asymptote)

(defcustom pdf-view-command
  (cond
   ((eq window-system 'x)
    "xpdf")
   ((eq window-system 'w32)
    "acroread")
   (t "_indefinite_pdf_viewer"))
  "Command to view a Postscript file."
  :type 'file
  :group 'asymptote)

(defcustom lasy-keep-key t
  "*If on, keep the binding key of lasy-mode in all latex-mode.
The folowing keys are added:
\(kbd \"<M-return>\"\) `lasy-view-pdf-via-pdflatex\)
\(kbd \"<C-return>\"\) `lasy-view-ps\)
\(kbd \"<C-M-return>\"\) `lasy-view-pdf-via-ps2pdf\)"
  :type 'boolean
  :group 'asymptote)

(defvar asy-TeX-master-file nil
  "TeX file associate with current asymptote code.
This variable must be modified only using the function 'asy-set-master-tex by M-x asy-set-master-tex <RET>.")
(make-variable-buffer-local 'asy-TeX-master-file)

(defvar asy-command-name
  "\\<abort\\>\\|\\<abs\\>\\|\\<acos\\>\\|\\<aCos\\>\\|\\<acosh\\>\\|\\<add\\>\\|\\<addSaveFunction\\>\\|\\<alias\\>\\|\\<align\\>\\|\\<all\\>\\|\\<angle\\>\\|\\<arc\\>\\|\\<ArcArrow\\>\\|\\<ArcArrows\\>\\|\\<arcarrowsize\\>\\|\\<arcdir\\>\\|\\<arclength\\>\\|\\<arcpoint\\>\\|\\<arctime\\>\\|\\<arrow\\>\\|\\<Arrow\\>\\|\\<arrow2\\>\\|\\<arrowhead\\>\\|\\<arrowheadbbox\\>\\|\\<Arrows\\>\\|\\<arrowsize\\>\\|\\<asin\\>\\|\\<aSin\\>\\|\\<asinh\\>\\|\\<ask\\>\\|\\<assert\\>\\|\\<asy\\>\\|\\<atan\\>\\|\\<aTan\\>\\|\\<atan2\\>\\|\\<atanh\\>\\|\\<atbreakpoint\\>\\|\\<atexit\\>\\|\\<atexit)\\>\\|\\<attach\\>\\|\\<AvantGarde\\>\\|\\<axialshade\\>\\|\\<azimuth\\>\\|\\<bar\\>\\|\\<Bar\\>\\|\\<Bars\\>\\|\\<barsize\\>\\|\\<basealign\\>\\|\\<baseline\\>\\|\\<bbox\\>\\|\\<BeginArcArrow\\>\\|\\<BeginArrow\\>\\|\\<BeginBar\\>\\|\\<beginclip\\>\\|\\<BeginDotMargin\\>\\|\\<begingroup\\>\\|\\<BeginMargin\\>\\|\\<BeginPenMargin\\>\\|\\<beginpoint\\>\\|\\<Blank\\>\\|\\<Bookman\\>\\|\\<box\\>\\|\\<breakpoint\\>\\|\\<breakpoints\\>\\|\\<buildRestoreDefaults)\\>\\|\\<buildRestoreThunk)\\>\\|\\<cap\\>\\|\\<cast\\>\\|\\<cbrt\\>\\|\\<cd\\>\\|\\<ceil\\>\\|\\<Ceil\\>\\|\\<circle\\>\\|\\<clear\\>\\|\\<clip\\>\\|\\<close\\>\\|\\<cmyk\\>\\|\\<colatitude\\>\\|\\<colorless\\>\\|\\<colors\\>\\|\\<complement\\>\\|\\<conj\\>\\|\\<controls\\>\\|\\<cos\\>\\|\\<Cos\\>\\|\\<cosh\\>\\|\\<Courier\\>\\|\\<_cputime\\>\\|\\<cputime\\>\\|\\<cross\\>\\|\\<csv\\>\\|\\<ct\\>\\|\\<cubiclength\\>\\|\\<cubicroots\\>\\|\\<curl\\>\\|\\<cycle\\>\\|\\<cyclic\\>\\|\\<debugger\\>\\|\\<deconstruct\\>\\|\\<defaultpen\\>\\|\\<degrees\\>\\|\\<Degrees\\>\\|\\<determinant\\>\\|\\<dimension\\>\\|\\<dir\\>\\|\\<dirtime\\>\\|\\<dot\\>\\|\\<DotMargin\\>\\|\\<DotMargins\\>\\|\\<dotsize\\>\\|\\<Dotted\\>\\|\\<_draw\\>\\|\\<draw\\>\\|\\<Draw\\>\\|\\<ecast\\>\\|\\<ellipse\\>\\|\\<empty\\>\\|\\<EndArcArrow\\>\\|\\<EndArrow\\>\\|\\<EndBar\\>\\|\\<endclip\\>\\|\\<EndDotMargin\\>\\|\\<endgroup\\>\\|\\<endl\\>\\|\\<EndMargin\\>\\|\\<EndPenMargin\\>\\|\\<endpoint\\>\\|\\<eof\\>\\|\\<eol\\>\\|\\<erase\\>\\|\\<erf\\>\\|\\<erfc\\>\\|\\<error\\>\\|\\<_eval\\>\\|\\<eval\\>\\|\\<exitfunction\\>\\|\\<exp\\>\\|\\<expi\\>\\|\\<fabs\\>\\|\\<fill\\>\\|\\<Fill\\>\\|\\<filldraw\\>\\|\\<FillDraw\\>\\|\\<fillrule\\>\\|\\<filltype\\>\\|\\<find\\>\\|\\<finite\\>\\|\\<firstcut\\>\\|\\<fixedscaling\\>\\|\\<floor\\>\\|\\<Floor\\>\\|\\<flush\\>\\|\\<fmod\\>\\|\\<font\\>\\|\\<fontcommand\\>\\|\\<fontsize\\>\\|\\<format\\>\\|\\<gamma\\>\\|\\<getc\\>\\|\\<getint\\>\\|\\<getpair\\>\\|\\<getreal\\>\\|\\<getstring\\>\\|\\<gouraudshade\\>\\|\\<graphic\\>\\|\\<gray\\>\\|\\<grestore\\>\\|\\<gsave\\>\\|\\<gui\\>\\|\\<GUI\\>\\|\\<GUIop\\>\\|\\<GUIreset\\>\\|\\<Helvetica\\>\\|\\<hypot\\>\\|\\<identity\\>\\|\\<image\\>\\|\\<init\\>\\|\\<initdefaults\\>\\|\\<input\\>\\|\\<insert\\>\\|\\<inside\\>\\|\\<interact\\>\\|\\<interp\\>\\|\\<intersect\\>\\|\\<intersectionpoint\\>\\|\\<inverse\\>\\|\\<invisible\\>\\|\\<italic\\>\\|\\<label\\>\\|\\<Label\\>\\|\\<labelmargin\\>\\|\\<labels\\>\\|\\<Landscape\\>\\|\\<lastcut\\>\\|\\<latitude\\>\\|\\<latticeshade\\>\\|\\<layer\\>\\|\\<legend\\>\\|\\<length\\>\\|\\<line\\>\\|\\<linecap\\>\\|\\<linejoin\\>\\|\\<lineskip\\>\\|\\<linetype\\>\\|\\<linewidth\\>\\|\\<locatefile\\>\\|\\<location\\>\\|\\<log\\>\\|\\<log10\\>\\|\\<longitude\\>\\|\\<makedraw\\>\\|\\<makepen\\>\\|\\<map\\>\\|\\<margin\\>\\|\\<Margin\\>\\|\\<Margins\\>\\|\\<marginT\\>\\|\\<Mark\\>\\|\\<marker\\>\\|\\<marknodes\\>\\|\\<markroutine\\>\\|\\<markuniform\\>\\|\\<math\\>\\|\\<max\\>\\|\\<maxbound\\>\\|\\<maxcoords\\>\\|\\<merge\\>\\|\\<MidArcArrow\\>\\|\\<MidArrow\\>\\|\\<midpoint\\>\\|\\<min\\>\\|\\<minbound\\>\\|\\<minipage\\>\\|\\<NewCenturySchoolBook\\>\\|\\<newpage\\>\\|\\<nib\\>\\|\\<NoFill\\>\\|\\<NoMargin\\>\\|\\<none\\>\\|\\<None\\>\\|\\<nullexitfcn\\>\\|\\<orientation\\>\\|\\<output\\>\\|\\<overwrite\\>\\|\\<pack\\>\\|\\<Palatino\\>\\|\\<pattern\\>\\|\\<pause\\>\\|\\<pen)\\>\\|\\<pen))\\>\\|\\<Pen\\>\\|\\<PenMargin\\>\\|\\<PenMargins\\>\\|\\<piecewisestraight\\>\\|\\<point\\>\\|\\<polar\\>\\|\\<polygon\\>\\|\\<Portrait\\>\\|\\<postcontrol\\>\\|\\<postscript\\>\\|\\<pow10\\>\\|\\<precision\\>\\|\\<precontrol\\>\\|\\<prepend\\>\\|\\<quadraticroots\\>\\|\\<quotient\\>\\|\\<radialshade\\>\\|\\<RadialShade\\>\\|\\<radians\\>\\|\\<rand\\>\\|\\<read1\\>\\|\\<read2\\>\\|\\<read3\\>\\|\\<readGUI\\>\\|\\<readline\\>\\|\\<realmult\\>\\|\\<rectify\\>\\|\\<reflect\\>\\|\\<relative\\>\\|\\<Relative\\>\\|\\<relativedistance\\>\\|\\<reldir\\>\\|\\<relpoint\\>\\|\\<reltime\\>\\|\\<remainder\\>\\|\\<replace\\>\\|\\<resetdefaultpen\\>\\|\\<restore\\>\\|\\<restoredefaults\\>\\|\\<reverse\\>\\|\\<rfind\\>\\|\\<rgb\\>\\|\\<rotate\\>\\|\\<round\\>\\|\\<Round\\>\\|\\<save)\\>\\|\\<savedefaults)\\>\\|\\<scale\\>\\|\\<scroll\\>\\|\\<search\\>\\|\\<Seascape\\>\\|\\<seconds\\>\\|\\<seek\\>\\|\\<sequence\\>\\|\\<sgn\\>\\|\\<shift\\>\\|\\<shiftless\\>\\|\\<shipout\\>\\|\\<sin\\>\\|\\<Sin\\>\\|\\<single\\>\\|\\<sinh\\>\\|\\<size\\>\\|\\<slant\\>\\|\\<solve\\>\\|\\<sort\\>\\|\\<sourceline\\>\\|\\<spec\\>\\|\\<sqrt\\>\\|\\<srand\\>\\|\\<stop\\>\\|\\<straight\\>\\|\\<string\\>\\|\\<subpath\\>\\|\\<substr\\>\\|\\<suffix\\>\\|\\<sum\\>\\|\\<Symbol\\>\\|\\<system\\>\\|\\<tab\\>\\|\\<tan\\>\\|\\<Tan\\>\\|\\<tanh\\>\\|\\<tell\\>\\|\\<tension\\>\\|\\<tex\\>\\|\\<texify\\>\\|\\<TeXify\\>\\|\\<texpreamble\\>\\|\\<texreset\\>\\|\\<time\\>\\|\\<TimesRoman\\>\\|\\<triangulate\\>\\|\\<tridiagonal\\>\\|\\<trim\\>\\|\\<TrueMargin\\>\\|\\<truepoint\\>\\|\\<unfill\\>\\|\\<UnFill\\>\\|\\<unit\\>\\|\\<UpsideDown\\>\\|\\<uptodate\\>\\|\\<usepackage\\>\\|\\<usersetting\\>\\|\\<VERSION\\>\\|\\<write\\>\\|\\<xinput\\>\\|\\<xoutput\\>\\|\\<xpart\\>\\|\\<xscale\\>\\|\\<xtrans\\>\\|\\<ypart\\>\\|\\<yscale\\>\\|\\<ytrans\\>\\|\\<ZapfChancery\\>\\|\\<ZapfDingbats\\>\\|\\<zpart\\>")

(defvar asy-constant-name
  "\\<infinity\\>\\|\\<Infinity\\>\\|\\<inches\\>\\|\\<inch\\>\\|\\<cm\\>\\|\\<mm\\>\\|\\<bp\\>\\|\\<pt\\>\\|\\<I\\>\\|\\<up\\>\\|\\<down\\>\\|\\<right\\>\\|\\<left\\>\\|\\<E\\>\\|\\<N\\>\\|\\<W\\>\\|\\<S\\>\\|\\<NE\\>\\|\\<NW\\>\\|\\<SW\\>\\|\\<SE\\>\\|\\<ENE\\>\\|\\<NNE\\>\\|\\<NNW\\>\\|\\<WNW\\>\\|\\<WSW\\>\\|\\<SSW\\>\\|\\<SSE\\>\\|\\<ESE\\>\\|\\<defaultfilename\\>\\|\\<Above\\>\\|\\<Below\\>\\|\\<stdin\\>\\|\\<stdout\\>\\|\\<unitsquare\\>\\|\\<unitcircle\\>\\|\\<circleprecision\\>\\|\\<invert\\>")

(defvar asy-user-command-name
  "\\<drawangle\\>\\|\\<drawline\\>\\|\\<drawrightangle\\>\\|\\<ccenter\\>\\|\\<ecenter\\>\\|\\<icenter\\>\\|.\\<labelA\\>\\|.labelB\\>\\|.labelC\\>\\|.Abc\\>\\|.abc\\>")

(font-lock-add-keywords 'asy-mode
                        `((,asy-command-name . font-lock-function-name-face)
                          (,asy-user-command-name . font-lock-function-name-face)
                          (,asy-constant-name . font-lock-constant-face)))

(if (locate-library "two-mode-mode") (require 'two-mode-mode) 
  (defvar two-mode-bool nil))

(require 'font-lock)
(require 'cc-mode)

(c-lang-defconst c-block-decls-with-vars
  "Keywords introducing declarations that can contain a block which
might be followed by variable declarations, e.g. like \"foo\" in
\"class Foo { ... } foo;\".  So if there is a block in a declaration
like that, it ends with the following ';' and not right away.

The keywords on list are assumed to also be present on one of the
`*-decl-kwds' lists."
  t        nil
  objc '("union" "enum" "typedef") ;; Asymptote doesn't require ';' after struct
  c '("struct" "union" "enum" "typedef")
  c++      '("class" "struct" "union" "enum" "typedef"))

(setq mode-name "Asymptote")
(if (featurep 'xemacs)
    (turn-on-font-lock)
  (global-font-lock-mode t))
(column-number-mode t)

(defun asy-get-temp-file-name()
  "Get a temp file name for printing."
  (make-temp-file
   (expand-file-name "asy" asy-temp-dir) nil ".asy"))


(defun asy-self-compile-view(Filename)
  "Compile Asymptote code Filename and view compilation result with the function 'shell-command'."
  (interactive)
  (let*
      ((buffer-base-name (file-name-sans-extension  buffer-file-name))
       (asy-command 
	(concat asy-command-location
		"asy -V " Filename
                " 2>" buffer-base-name "_asy.log")))
    (shell-command asy-command)
    (set-window-text-height nil (ceiling (* (frame-height) .7)))))

(defun asy-compile-view()
  "Compile Asymptote code and view compilation result."
  (interactive)
  (if (and (boundp two-mode-bool) two-mode-bool)
      (lasy-compile-view)
    (progn
      (let*
          ((buffer-base-name (file-name-sans-extension (file-name-nondirectory buffer-file-name)))
           (asy-command (concat asy-command-location
                                "asy -V "
                                buffer-base-name
                                ".asy 2>"
                                buffer-base-name "_asy.log")))
        (if (buffer-modified-p) (save-buffer))
        (shell-command asy-command)
        (asy-error-message t)
        ))))

(defun asy-error-message(&optional P)
  (let ((asy-last-error
         (asy-log-field-string
          (concat (file-name-sans-extension buffer-file-name) "_asy.log") 0)))
    (if (and asy-last-error (not (string= asy-last-error "")))
        (message (concat asy-last-error (if P "\nPress F4 to goto to error" "")))
      )
    ))

(defun asy-log-field-string(Filname Field)
  "Return field of last line of file filaname.
Fields are define as 'field1:field2.field3:field4' . Field=0 <-> all fields"
  (with-temp-buffer
    (progn
      (insert-file Filname)
      (beginning-of-buffer)
      (if (re-search-forward "^\\(.*?\\): \\(.*?\\)\\.\\(.*?\\):\\(.*\\)$" (point-max) t)
          (progn
            (beginning-of-buffer)
            (while (re-search-forward "^\\(.*?\\): \\(.*?\\)\\.\\(.*?\\):\\(.*\\)$" (point-max) t))
            (match-string Field))
        nil))))

(defun asy-goto-error()
  "Go to point of last error within asy/lasy-mode."
  (interactive)
  (let* ((log-file (concat (file-name-sans-extension (buffer-file-name)) "_asy.log"))
         (line_ (asy-log-field-string log-file 2)))
    (if line_
        (if (and (boundp two-mode-bool) two-mode-bool)
            (progn
              (re-search-backward "\\\\begin{asy")
              (next-line (1- (string-to-number line_)))
              (asy-error-message))
          (progn
            (goto-line (string-to-number line_))
            (asy-error-message)))
      (progn (message "No error"))
      )))


(defun asy-grep (Regexp)
  "Internal function used by asymptote."
  (let ((Strout ""))
    (progn
      (beginning-of-buffer)
      (while (re-search-forward Regexp (point-max) t)
        (setq Strout (concat Strout (match-string 0) "\n\n")))
      (if (string= Strout "") "No match.\n" Strout))))


(defun asy-show-command-at-point()
  "Show the Asymptote definitions of the command at point."
  (interactive)
  (save-excursion
    (let ((cWord (current-word))
          (cWindow (selected-window)))
      (switch-to-buffer-other-window "*asy-help*")
      (call-process-shell-command
       (concat asy-command-location "asy -l") nil t nil)
      (let ((rHelp (asy-grep (concat "^.*\\b" cWord "(\\(.\\)*?$"))))
        (erase-buffer)
        (insert rHelp))
      (asy-mode)
      (use-local-map nil)
      (goto-char (point-min))
      (select-window cWindow))))



;;;;;;; TEMPO ;;;;;;;;;;;;;;;
(defvar asy-tempo-tags nil
  "Tempo tags for ASY mode")

;;; ASY-Mode Templates
(require 'tempo)
(require 'advice)
(setq-default tempo-interactive t)
(setq-default 
 tempo-match-finder "\\b\\([^\b]+\\)\\=")  ;; The definition in tempo.el is false.
(tempo-use-tag-list 'asy-tempo-tags)

;;; Function to construct asy commands
(defun asy-tempo (l)
  "Construct tempo-template for Asymptote commands."
  (let* ((tag (car l))
	 (element (nth 1 l)))
    (tempo-define-template tag element tag nil 'asy-tempo-tags)))

;;; Templates appended to asy-tempo-tags
(mapcar
 'asy-tempo
 '(("dir"  ("{dir(" p ")}"))
   ("intp"  ("intersectionpoint(" p ", " p ")"))
   ("intt"  ("intersectiontime(" p ", " p ")"))
   ("rot"  ("rotate(" p ")*"))
   ("shi"  ("shift(" p ")*"))
   ("sca"  ("scale(" p ")*"))
   ("xsc"  ("xscale(" p ")*"))
   ("ysc"  ("yscale(" p ")*"))
   ("zsc"  ("zscale(" p ")*"))
   ("if"  (& >"if (" p ")"n>
             "{"n>
             r n >
             "}"> %))
   ("els"  (& >"else" n>
              "{"n>
              r n >
              "}"> %))
   ("ife"  (& >"if (" p ")" n>
              "{" n>
              r n
              "}" > n> 
              "else" n>
              "{" n>
              p n>
              "}" > %))
   ("for"  (& >"for (" p " ," p " ," p ")" n>
              "{" n>
              r n >
              "}"> %))
   ("whi"  (& >"while (" p ")" n>
              "{" n>
              r n>
              "}"> %))
   
   ))

(add-hook 'asy-mode-hook 
	  '(lambda ()
	     (tempo-use-tag-list 'asy-tempo-tags)
             (if (boundp flyspell-mode) (flyspell-mode -1))
	     ))

;; Definition of insertion functions
(defun dir()
  (interactive)
  (insert "{dir()}")
  (forward-char -2))

;;; ************************************
;;; asy-mode mixed with LaTeX-mode: lasy
;;; ************************************

(defun lasy-mode ()
  "Treat, in some cases, the current buffer as a literate Asymptote program."
  (interactive)
  (setq default-mode    '("LaTeX" latex-mode)
        second-modes     '(("Asymptote"
                            "\\begin{asy"
                            "\\end{asy"
                            asy-mode)))
  (if two-mode-bool
      (progn
        (latex-mode)
        (if lasy-keep-key
            (use-local-map lasy-mode-map)
          (use-local-map LaTeX-mode-map)))
    (progn
      (two-mode-mode))))

(add-hook 'two-mode-hook
          '(lambda ()
             (if (string= (downcase (substring mode-name 0 5)) "latex")
                 (use-local-map lasy-mode-map)
               )
             ))
(add-hook 'two-mode-switch-hook
          '(lambda ()
             (if (string= (downcase (substring mode-name 0 5)) "latex")
                 (use-local-map lasy-mode-map)
               )
             ))

(defun lasy-compile-view()
  "Compile region at point between \\begin{asy} and \\end{asy}"
  (interactive)
  (save-excursion
    (let ((Filename (asy-get-temp-file-name)))
      (re-search-forward "\\\\end{asy}")
      (re-search-backward "\\\\begin{asy}\\(\n\\|.\\)*?\\\\end{asy}")
      (write-region (match-string 0) 0 Filename)
      (with-temp-file Filename
        (insert-file Filename)
        (beginning-of-buffer)
        (while (re-search-forward "\\\\begin{asy}\\|\\\\end{asy}" (point-max) t)
          (replace-match "")))
      (asy-self-compile-view Filename)
      )))

(defun asy-set-master-tex ()
  "Set the local variable 'asy-TeX-master-file.
This variable is used by 'asy-master-tex-view-ps"
  (interactive)
  (set (make-local-variable 'asy-TeX-master-file)
       (file-name-sans-extension
        (file-relative-name
         (expand-file-name
          (read-file-name "TeX document: ")))))
  (if (string= (concat default-directory asy-TeX-master-file)
               (file-name-sans-extension buffer-file-name))
      (prog1
          (set (make-local-variable 'asy-TeX-master-file) nil)
        (error "You should never give the same name to the TeX file and Asymptote file"))
    (save-excursion
      (end-of-buffer)
      (if (re-search-backward "asy-TeX-master-file\\(.\\)*$" 0 t)
          (replace-match (concat "asy-TeX-master-file: \"" asy-TeX-master-file "\""))
        (insert (concat "
/// Local Variables:
/// asy-TeX-master-file: \"" asy-TeX-master-file "\"
/// End:")) t))))

(defun asy-master-tex-error ()
  "Asy-mode internal use..."
  (if (y-or-n-p "You try to compile the TeX document that contains this picture.
You must set the local variables asy-TeX-master-file.
Do you want set this variable now ?")
      (asy-set-master-tex) nil))

(defun asy-master-tex-view (Func-view &optional Force)
  "Compile the LaTeX document that contains the picture of the current Asymptote code with the function Func-view.
Func-view can be one of 'lasy-view-ps, 'lasy-view-pdf-via-pdflatex, 'lasy-view-pdf-via-ps2pdf."
  (interactive)
  (if (and (boundp two-mode-bool) two-mode-bool)
      (progn  ;;Current mode is lasy-mode no asy-mode
        (funcall Func-view Force))
    (if asy-TeX-master-file
        (if (string= asy-TeX-master-file
                     (file-name-sans-extension buffer-file-name))
            (error "You should never give the same name to the TeX file and Asymptote file")
          (funcall Func-view  Force asy-TeX-master-file))
      (if (asy-master-tex-error)
          (funcall Func-view Force asy-TeX-master-file)))))

(defun lasy-view-ps (&optional Force  Filename )
  "Compile a LaTeX document embedding Asymptote code with latex->asy->latex->dvips and/or view the Poscript output.
If optional argument Force is t then force compilation."
  (interactive)
  (if (buffer-modified-p) (save-buffer))
  (let
      ((b-b-n  (if Filename Filename (file-name-sans-extension buffer-file-name))))
    (if (or (file-newer-than-file-p
             (concat b-b-n ".tex")
             (concat b-b-n ".ps"))
            Force)
	(let
	    ((asy-command 
	      (concat  "latex " b-b-n ".tex && "
                       "{ if [[ -f " b-b-n ".asy ]]; then " asy-command-location "asy " b-b-n ".asy && latex " b-b-n ".tex; else true; fi;} && "
                       "dvips " b-b-n ".dvi -o " b-b-n ".ps && " ps-view-command " " b-b-n ".ps")))
	  (shell-command asy-command)
	  (set-window-text-height nil (ceiling (* (frame-height) .7))))
      (let
	  ((asy-command 
	    (format  "%s %s.ps" ps-view-command b-b-n)))
	(shell-command asy-command))
      )))



(defun lasy-view-pdf-via-pdflatex (&optional Force Filename)
  "Compile a LaTeX document embedding Asymptote code with pdflatex->asy->pdflatex and/or view the PDF output.
If optional argument Force is t then force compilation."
  (interactive)
  (if (buffer-modified-p) (save-buffer))
  (let
      ((b-b-n  (if Filename Filename (file-name-sans-extension buffer-file-name))))
    (if (or (file-newer-than-file-p
             (concat b-b-n ".tex")
             (concat b-b-n ".pdf"))
            Force)
	(let
	    ((asy-command 
	      (concat  "pdflatex -shell-escape " b-b-n ".tex && "
                       "{ if [[ -f " b-b-n ".asy ]]; then " asy-command-location "asy " b-b-n ".asy && pdflatex -shell-escape " b-b-n ".tex; else true; fi;} && "
		       pdf-view-command " " b-b-n ".pdf &")))
	  (shell-command asy-command)
	  (set-window-text-height nil (ceiling (* (frame-height) .7))))
      (let
	  ((asy-command 
	    (format  "%s %s.pdf" pdf-view-command b-b-n)))
	(shell-command asy-command))
      )))




(defun lasy-view-pdf-via-ps2pdf (&optional Force Filename)
  "Compile a LaTeX document embedding Asymptote code with latex->asy->latex->dvips->ps2pdf14 and/or view the PDF output.
If optional argument Force is t then force compilation."
  (interactive)
  (if (buffer-modified-p) (save-buffer))
  (let
      ((b-b-n  (if Filename Filename (file-name-sans-extension buffer-file-name))))
    (if (or (file-newer-than-file-p
             (concat b-b-n ".tex")
             (concat b-b-n ".pdf"))
            Force)
	(let
	    ((asy-command 
	      (concat  "latex " b-b-n" .tex && "
                       "{ if [[ -f " b-b-n ".asy ]]; then " asy-command-location "asy " b-b-n ".asy && latex " b-b-n ".tex; else true; fi;} && "
		       "dvips  -Ppdf -ta4 " b-b-n ".dvi -o " b-b-n ".ps && "
		       "ps2pdf14 -dPDFSETTINGS=/prepress -dAutoFilterGrayImages=false -dAutoFilterColorImages=false -dColorImageFilter=/FlateEncode -dGrayImageFilter=/FlateEncode -dAutoRotatePages=/None " b-b-n ".ps " b-b-n ".pdf && "
		       pdf-view-command " " b-b-n ".pdf")))
	  (shell-command asy-command)
	  (set-window-text-height nil (ceiling (* (frame-height) .7))))
      (let
	  ((asy-command 
	    (format  "%s %s.pdf" pdf-view-command b-b-n)))
	(shell-command asy-command))
      )))


(eval-after-load "latex"
  '(progn
     ;; Hack maybe not totally safe, I don't know.
     ;; (add-to-list 'TeX-expand-list
     ;;              '("%a"
     ;;                (lambda nil
     ;;                  asy-command-location)) t)
     
     ;; (add-to-list 'TeX-command-list
     ;;              '("asy-LaTeX" "%l \"%(mode)\\input{%t}\" && %aasy %s.asy && %l \"%(mode)\\input{%t}\" && %V"
     ;;                TeX-run-interactive nil (latex-mode)
     ;;                :help "Run LaTeX && Asymptote && LaTeX\n
     ;; Be sure to have\n
     ;; \usepackage{graphicx}\n
     ;; \usepackage{asymptote}"))
     
     ;; (add-to-list 'TeX-command-list
     ;;               '("asy-pdflaTex" "pdflatex -shell-escape %t && %aasy %s.asy && pdflatex -shell-escape %t"
     ;;                TeX-run-command nil (latex-mode)
     ;;                :help "Run pdflatex && Asymptote && pdflatex\n
     ;; Be sure to have\n
     ;; \usepackage{graphicx}\n
     ;; \usepackage{epstopdf}\n
     ;; \usepackage{asymptote}"))
     
     ;; (add-to-list 'TeX-command-list
     ;;               '("asy-ps" "%l \"%(mode)\\input{%t}\" && %aasy %s.asy && %(o?)dvips %d -o %f"
     ;;                TeX-run-command nil (latex-mode)
     ;;                :help "Run LaTeX && Asymptote && LaTeX\n
     ;; Be sure to have\n
     ;; \usepackage{graphicx}\n
     ;; \usepackage{asymptote}"))
     
     ;; (add-to-list 'TeX-command-list
     ;;              '("asy-dvips-pdf" "%l \"%(mode)\\input{%t}\" && %aasy %s.asy && %(o?)dvips %d -o %f && ps2df14 -dPDFSETTINGS=/prepress -dAutoFilterGrayImages=false -dAutoFilterColorImages=false -dColorImageFilter=/FlateEncode -dGrayImageFilter=/FlateEncode -dAutoRotatePages=/None %f %s.pdf"
     ;;                TeX-run-command nil (latex-mode)
     ;;                :help "Run LaTeX && Asymptote && LaTeX && dvips && ps2pdf14\n
     ;; Be sure to have\n
     ;; \usepackage{graphicx}\n
     ;; \usepackage{asymptote}"))
     
     (if (locate-library "texmathp")
         (prog1
             ;; Not necessary but it's very useful.
             (require 'texmathp)
           (define-key LaTeX-mode-map [(^)] #'(lambda ()
                                                (interactive)
                                                (if (texmathp)
                                                    (progn
                                                      (insert "^{}")
                                                      (backward-char))
                                                  (insert "^"))))
           
           (define-key LaTeX-mode-map [(_)] #'(lambda ()
                                                (interactive)
                                                (if (texmathp)
                                                    (progn
                                                      (insert "_{}")
                                                      (backward-char))
                                                  (insert "_"))))
           
           (define-key LaTeX-mode-map (kbd "²") #'(lambda ()
                                                    (interactive)
                                                    (insert "\\")))
           )
       (progn
         (message "texmathp not find...")))
     
     
     (setq lasy-mode-map (copy-keymap LaTeX-mode-map))
     
     (define-key lasy-mode-map (kbd "<M-return>") `lasy-view-pdf-via-pdflatex)
     (define-key lasy-mode-map (kbd "<M-S-return>")
       '(lambda ()
          (interactive)
          (lasy-view-pdf-via-pdflatex t)))
     (define-key lasy-mode-map (kbd "<C-return>") `lasy-view-ps)
     (define-key lasy-mode-map (kbd "<C-S-return>")
       '(lambda ()
          (interactive)
          (lasy-view-ps t)))
     (define-key lasy-mode-map (kbd "<C-M-return>") `lasy-view-pdf-via-ps2pdf)
     (define-key lasy-mode-map (kbd "<C-M-S-return>")
       '(lambda ()
          (interactive)
          (lasy-view-pdf-via-ps2pdf t)))
     (if lasy-keep-key (setq LaTeX-mode-map (copy-keymap lasy-mode-map)))
     ))

;; Goto to the forward/backward tempo's mark
(define-key asy-mode-map [M-right] 'tempo-forward-mark)
(define-key asy-mode-map [M-left]  'tempo-backward-mark)
;; Complete the tempo tag (the first three letters of a keyword)
(define-key asy-mode-map [f3] 'tempo-complete-tag)

;; Goto error of last compilation
(define-key asy-mode-map  (kbd "<f4>") 'asy-goto-error)

;; Save and compile the file with option -V
(define-key asy-mode-map  (kbd "C-c C-c") 'asy-compile-view)

;; Show the definitions of command at point
(define-key asy-mode-map  (kbd "C-c ?") 'asy-show-command-at-point)


(define-key asy-mode-map (kbd "<C-return>")
  '(lambda ()
     (interactive)
     (asy-master-tex-view 'lasy-view-ps)))
(define-key asy-mode-map (kbd "<C-S-return>")
  '(lambda ()
     (interactive)
     (asy-master-tex-view 'lasy-view-ps t)))

(define-key asy-mode-map (kbd "<M-return>")
  '(lambda ()
     (interactive)
     (asy-master-tex-view 'lasy-view-pdf-via-pdflatex)))

(define-key asy-mode-map (kbd "<M-S-return>")
  '(lambda ()
     (interactive)
     (asy-master-tex-view 'lasy-view-pdf-via-pdflatex t)))

(define-key asy-mode-map (kbd "<C-M-return>")
  '(lambda ()
     (interactive)
     (asy-master-tex-view 'lasy-view-pdf-via-ps2pdf)))

(define-key asy-mode-map (kbd "<C-M-S-return>")
  '(lambda ()
     (interactive)
     (asy-master-tex-view 'lasy-view-pdf-via-ps2pdf t)))


(provide `asy-mode)
