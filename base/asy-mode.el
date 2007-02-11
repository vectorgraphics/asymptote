;;; asy-mode.el

;; Copyright (C) 2006
;; Author: Philippe IVALDI 20 August 2006
;; Modified by: John Bowman 01 September 2006
;; Last modification: 06 February 2007 (Philippe Ivaldi)
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

;; INSTALLATION:
;; Place this file (asy-mode.el) and asy-keywords.el in your Emacs load path.
;; Then choose ONE of the following installation methods:

;; * Method 1:
;;   Copy and uncomment the following lines to your .emacs initialization file:
;;(autoload 'asy-mode "asy-mode.el" "Asymptote major mode." t)
;;(autoload 'lasy-mode "asy-mode.el" "hybrid Asymptote/Latex major mode." t)
;;(autoload 'asy-insinuate-latex "asy-mode.el" "Asymptote insinuate LaTeX." t)
;;(add-to-list 'auto-mode-alist '("\\.asy$" . asy-mode))

;; * Method 2:
;;   Copy and uncomment the following line to your .emacs initialization file:
;;(require 'asy-mode)

;; Notes:
;;
;; For full functionality the two-mode-mode package should also be installed
;; (http://www.dedasys.com/freesoftware/files/two-mode-mode.el).
;;
;; See also paragraph II of the documentation below to automate asy-insinuate-latex. 

;;;###autoload
(define-derived-mode asy-mode objc-mode "Asymptote"
  "Emacs mode for editing Asymptote source code.
For full functionality the `two-mode-mode' package should also be installed
(http://www.dedasys.com/freesoftware/files/two-mode-mode.el).

I. This package provides two modes:
1- asy-mode:
All the files with the extension '.asy' are edited in this mode, which provides the following features:
* Syntax color highlighting;
* Compiling and viewing current buffer with the key binding C-c C-c;
* Moving cursor to the error by pressing the key F4.
* Showing the available function prototypes for the command at the cursor with the key binding C-c ?
* Compiling and viewing a TeX document linked with the current buffer (usually a document that includes the output picture).
To link a Tex document try 'M-x asy-set-master-tex' follow by C-Return (see descriptions further of the key binding C-Return, C-S-Return, M-Return, M-S-Return etc within 2- lasy-mode)

2- lasy-mode
Editing a TeX file that contains Asymptote code is facilitated with the hybrid mode 'lasy-mode'.
Toggle lasy-mode with M-x lasy-mode.
In this hybrid mode the major mode is LaTeX when the cursor is in LaTeX code and becomes asy-mode when the cursor is between '\begin{asy}' and '\end{asy}'.
All the features of asy-mode are provided and the key binding C-c C-c of asy-mode compiles and views only the code of the picture where the cursor is.
Note that some keys binding are added to the LaTeX-mode-map in lasy-mode if the value of the variable lasy-extra-key is t (the default)
.
* C-return: compile (if the buffer/file is modified) and view the postscript output with sequence [latex->[asy->latex]->dvips]->PSviewer
* M-return: same with pdf output and with the sequence [pdflatex -shell-escape->[asy->pdflatex -shell-escape]]->PDFviewer
* C-M-return: same with pdf output and with the sequence [latex->[asy->latex]->dvips->ps2pdf]->PSviewer
* Add the Shift key to the sequence of keys to compile even if the file is not modified.

II. To add a menu bar in current 'latex-mode' buffer and activate hot keys, use 'M-x asy-insinuate-latex <RET>'.
You can automate this feature for all the 'latex-mode' buffers by inserting the five following lines in your .emacs initialization file:
  (eval-after-load \"latex\"
    '(progn
       ;; Add here your personal features for 'latex-mode':
       (asy-insinuate-latex t) ;; Asymptote globally insinuates Latex.
       ))

You can access this help within Emacs by the key binding C-h f asy-mode <RET>

BUGS:
This package has been created and tested with:
- Linux Debian Sarge
- GNU Emacs 22.0.50.1
- AUCTeX 11.55
- Asymptote 1.13

This package seems to work with XEmacs 21.4 but not all the features are available (in particular syntax highlighting).

Report bugs to http://asymptote.sourceforge.net

Some variables can be customized: M-x customize-group <RET> asymptote <RET>."

  (setq c++-font-lock-extra-types (cons "true" c++-font-lock-extra-types)))

(require 'font-lock)
(require 'cc-mode)
(require 'cl) ;; Common Lisp extensions for Emacs
(require 'compile)

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.asy$" . asy-mode))

(defvar running-xemacs-p (string-match "XEmacs\\|Lucid" emacs-version))

(when running-xemacs-p
  (defalias 'turn-on-font-lock-if-enabled 'ignore)
  (defalias 'line-number-at-pos 'line-number)
  (defvar temporary-file-directory (temp-directory)))

(when (or (< emacs-major-version 22) running-xemacs-p)
  ;; Add regexp for parsing the compilation errors of asy
  (add-to-list 'compilation-error-regexp-alist
               '("\\(.*?.asy\\): \\(.*?\\)\\.\\(.*?\\):" 1 2 3)))

(when (< emacs-major-version 22)
  (defun line-number-at-pos (&optional pos)
    "Return (narrowed) buffer line number at position POS.
If POS is nil, use current buffer location.
Counting starts at (point-min), so the value refers
to the contents of the accessible portion of the buffer."
    (let ((opoint (or pos (point))) start)
      (save-excursion
        (goto-char (point-min))
        (setq start (point))
        (goto-char opoint)
        (forward-line 0)
        (1+ (count-lines start (point)))))))

(defcustom lasy-extra-key t
  "* If on, the folowing binding keys are added in lasy-mode :
     (define-key lasy-mode-map (kbd \"<C-return>\") 'lasy-view-ps)
     (define-key lasy-mode-map (kbd \"<C-S-return>\") 'asy-master-tex-view-ps-f)
     (define-key lasy-mode-map (kbd \"<M-return>\") 'lasy-view-pdf-via-pdflatex)
     (define-key lasy-mode-map (kbd \"<M-S-return>\") 'asy-master-tex-view-pdflatex-f)
     (define-key lasy-mode-map (kbd \"<C-M-return>\") 'lasy-view-pdf-via-ps2pdf)
     (define-key lasy-mode-map (kbd \"<C-M-S-return>\") 'asy-master-tex-view-ps2pdf-f)

If you also want this feature in pure latex-mode, you can set this variable to `nil' and add these lines in your .emacs:

(require 'asy-mode)
(eval-after-load \"latex\"
  '(progn
     (define-key LaTeX-mode-map (kbd \"<C-return>\") 'lasy-view-ps)
     (define-key LaTeX-mode-map (kbd \"<C-S-return>\") 'asy-master-tex-view-ps-f)
     (define-key LaTeX-mode-map (kbd \"<M-return>\") 'lasy-view-pdf-via-pdflatex)
     (define-key LaTeX-mode-map (kbd \"<M-S-return>\") 'asy-master-tex-view-pdflatex-f)
     (define-key LaTeX-mode-map (kbd \"<C-M-return>\") 'lasy-view-pdf-via-ps2pdf)
     (define-key LaTeX-mode-map (kbd \"<C-M-S-return>\") 'asy-master-tex-view-ps2pdf-f)))"
  :type 'boolean
  :group 'asymptote)

(defcustom asy-compilation-buffer 'none
  "* 'visible means keep compilation buffer visible ;
  'available means keep compilation buffer available in other buffer but not visible;
  'none means delete compilation buffer automaticly after a *successful* compilation.
  'never means delete compilation buffer automatically after any compilation.
If the value is 'never', the compilation process is `shell-command' with poor management of errors."
  :type '(choice (const visible) (const available) (const none) (const never))
  :group 'asymptote)

(defcustom lasy-ask-about-temp-compilation-buffer t
  "* If t, ask before visiting a temporary buffer of compilation."
  :type 'boolean
  :group 'asymptote)

(defcustom lasy-compilation-inline-auto-detection nil
  "* If t, lasy-mode detects automatically if the option 'inline' is passed to asymptote.sty.
In case of 'inline' option, the compilation of a figure separately of the document is processed by rebuilding the preamble and compiling it as a file '.tex' containing only this picture.
  If nil (the default), the compilation of a figure separately of the document is processed by building a file '.asy', without the features of the LaTeX preamble."
  :type 'boolean
  :group 'asymptote)

(defcustom asy-command-location ""
  "* If not in the path, you can put here the name of the directory containing Asy's binary files.
this variable must end in / (UNIX) or \ (MSWindows)."
  :type 'directory
  :group 'asymptote)

(defcustom asy-command "asy -V"
  "* Command invoked to compile a Asymptote file.
You can define the location of this command with the variable `asy-command-location'."
  :type 'string
  :group 'asymptote)

(defcustom lasy-command "asy"
  "* Command invoked to compile a Asymptote file generated compiling a .tex file.
You can define the location of this command with the variable `asy-command-location'."
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

(defvar asy-TeX-master-file nil
  "TeX file associate with current asymptote code.
This variable must be modified only using the function 'asy-set-master-tex by M-x asy-set-master-tex <RET>.")
(make-variable-buffer-local 'asy-TeX-master-file)

(defvar lasy-compile-tex nil
  "* Internal use. t if LaTeX compilation come from latex-mode.")

(when (fboundp 'font-lock-add-keywords)
  (if (< max-specpdl-size 2000) (setq max-specpdl-size 2000))
  (defun asy-add-function-keywords (function-keywords face-name)
    (let* ((keyword-list (mapcar #'(lambda (x)
                                     (symbol-name  x))
                                 function-keywords))
           (keyword-regexp (concat "\\<\\("
                                   (regexp-opt keyword-list)
                                   "\\)(")))
      (font-lock-add-keywords 'asy-mode
                              `((,keyword-regexp 1 ',face-name)))))
  
  (defun asy-add-variable-keywords (function-keywords face-name)
    (let* ((keyword-list (mapcar #'(lambda (x)
                                     (symbol-name  x))
                                 function-keywords))
           (keyword-regexp (concat "\\<[0-9]*\\("
                                   (regexp-opt keyword-list)
                                   "\\)\\(?:[^(a-zA-Z]\\|\\'\\)")))
      (font-lock-add-keywords 'asy-mode
                              `((,keyword-regexp 1 ',face-name)))))
  
  ;; External definitions of keywords:
  ;; asy-function-name and asy-variable-name
  (load-library "asy-keywords.el")
  
  (defcustom asy-extra-type-name '()
    "Extra user type names highlighted with 'font-lock-type-face"
    :type '(repeat symbol)
    :group 'asymptote)
  
  (defcustom asy-extra-function-name
    '()
    "Extra user function names highlighted with 'font-lock-function-name-face"
    :type '(repeat symbol)
    :group 'asymptote)
  
  (defcustom asy-extra-variable-name '()
    "Extra user variable names highlighted with 'font-lock-constant-face"
    :type '(repeat symbol)
    :group 'asymptote)
  
  (asy-add-variable-keywords
   asy-keyword-name 
   'font-lock-builtin-face)
  
  (asy-add-variable-keywords
   (nconc asy-type-name asy-extra-type-name)
   'font-lock-type-face)
  
  (asy-add-function-keywords
   (nconc asy-function-name asy-extra-function-name)
   'font-lock-function-name-face)
  
  (asy-add-variable-keywords
   (nconc asy-variable-name asy-extra-variable-name)
   'font-lock-constant-face)
  
  (defface asy-environment-face
    `((t
       (:underline t :inverse-video t)))
    "Face used to highlighting the keywords '\\begin{asy}' and '\\end{asy}' within lasy-mode."
    :group 'asymptote)
  
  (font-lock-add-keywords
   'asy-mode
   '(("\\\\begin{asy}.*" . 'asy-environment-face)
     ("\\\\end{asy}" . 'asy-environment-face)))
  )

(when (fboundp 'c-lang-defconst)
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
    c++      '("class" "struct" "union" "enum" "typedef")))

(setq buffers-menu-max-size nil)
(setq mode-name "Asymptote")

(if running-xemacs-p
    (defvar asy-menu
      '("Asy"
        ["Toggle lasy-mode"  lasy-mode :active (and (featurep 'two-mode-mode) two-mode-bool)]
        ["Compile/View"  asy-compile t]
        ["Go to error" asy-goto-error t]
        ["Describe command" asy-show-function-at-point t]"--"
        ("Master TeX file"
         ["Set/Change value" (asy-set-master-tex) :active (not (and (boundp two-mode-bool) two-mode-bool))]
         ["Erase value" (asy-unset-master-tex) :active (not (and (boundp two-mode-bool) two-mode-bool))]
         ("Compile OR View"
          ["PS"  asy-master-tex-view-ps :active t]
          ["PDF (pdflatex)" asy-master-tex-view-pdflatex :active t]
          ["PDF (ps2pdf)" asy-master-tex-view-ps2pdf :active t])
         ("Compile AND View"
          ["PS"  asy-master-tex-view-ps-f :active t]
          ["PDF (pdflatex)" asy-master-tex-view-pdflatex-f :active t]
          ["PDF (ps2pdf)" asy-master-tex-view-ps2pdf-f :active t]))
        ["Asymptote insinuates globally LaTeX"  asy-insinuate-latex-globally :active (not asy-insinuate-latex-globally-p)]"--"
        ("Debugger Buffer"
         ["Visible" (setq asy-compilation-buffer 'visible) :style radio :selected (eq asy-compilation-buffer 'visible) :active t]
         ["Available" (setq asy-compilation-buffer 'available)  :style radio :selected (eq asy-compilation-buffer 'available) :active t]
         ["None" (setq asy-compilation-buffer 'none)  :style radio :selected (eq asy-compilation-buffer 'none) :active t]
         ["Never" (setq asy-compilation-buffer 'never)  :style radio :selected (eq asy-compilation-buffer 'never) :active t])
        ("Compilation Options" :included (and (featurep 'two-mode-mode) two-mode-bool)
         ["Enable Automatic Detection of Option" (setq lasy-compilation-inline-auto-detection t) :style radio :selected lasy-compilation-inline-auto-detection :active t]
         ["Disable Automatic Detection of Option" (setq lasy-compilation-inline-auto-detection nil)  :style radio :selected (not lasy-compilation-inline-auto-detection) :active t])
        ["Customize" (customize-group "asymptote") :active t]
        ["Help" (describe-function 'asy-mode) :active t]
        ))
  (defvar asy-menu
    '("Asy"
      ["Toggle Lasy-Mode"  lasy-mode :visible (and (featurep 'two-mode-mode) two-mode-bool)]
      ["Compile/View"  asy-compile t]
      ["Go to Error" asy-goto-error t]
      ["Describe Command" asy-show-function-at-point t]"--"
      ("Master TeX File"
       ["Set/Change Value" (asy-set-master-tex) :active (not (and (boundp two-mode-bool) two-mode-bool)) :key-sequence nil]
       ["Erase Value" (asy-unset-master-tex) :active (not (and (boundp two-mode-bool) two-mode-bool)) :key-sequence nil]
       ("Compile or View"
        ["PS"  asy-master-tex-view-ps :active t]
        ["PDF (pdflatex)" asy-master-tex-view-pdflatex :active t]
        ["PDF (ps2pdf)" asy-master-tex-view-ps2pdf :active t])
       ("Compile and View"
        ["PS"  asy-master-tex-view-ps-f :active t]
        ["PDF (pdflatex)" asy-master-tex-view-pdflatex-f :active t]
        ["PDF (ps2pdf)" asy-master-tex-view-ps2pdf-f :active t]))
      ["Asymptote Insinuates Globally LaTeX"  asy-insinuate-latex-globally :active (not asy-insinuate-latex-globally-p)]"--"
      ("Debugger Buffer"
       ["Visible" (setq asy-compilation-buffer 'visible) :style radio :selected (eq asy-compilation-buffer 'visible) :active t :key-sequence nil]
       ["Available" (setq asy-compilation-buffer 'available)  :style radio :selected (eq asy-compilation-buffer 'available) :active t :key-sequence nil]
       ["None" (setq asy-compilation-buffer 'none)  :style radio :selected (eq asy-compilation-buffer 'none) :active t :key-sequence nil]
       ["Never" (setq asy-compilation-buffer 'never)  :style radio :selected (eq asy-compilation-buffer 'never) :active t :key-sequence nil])
      ("Compilation Options" :visible (and (featurep 'two-mode-mode) two-mode-bool)
       ["Enable Automatic Detection of Option" (setq lasy-compilation-inline-auto-detection t) :style radio :selected lasy-compilation-inline-auto-detection :active t :key-sequence nil]
       ["Disable Automatic Detection of Option" (setq lasy-compilation-inline-auto-detection nil)  :style radio :selected (not lasy-compilation-inline-auto-detection) :active t :key-sequence nil])
      ["Customize" (customize-group "asymptote") :active t :key-sequence nil]
      ["Help" (describe-function 'asy-mode) :active t :key-sequence nil]
      )))
(easy-menu-define asy-mode-menu asy-mode-map "Asymptote Mode Commands" asy-menu)
;; On the hook for XEmacs only.
(if running-xemacs-p
    (add-hook 'asy-mode-hook
              (lambda ()
                (and (eq major-mode 'asy-mode)
                     (easy-menu-add asy-mode-menu asy-mode-map)))))

(defun asy-get-temp-file-name(&optional noext)
  "Get a temp file name for printing."
  (if running-xemacs-p
      (concat (make-temp-name asy-temp-dir) (if noext "" ".asy"))
    (concat (make-temp-file
             (expand-file-name "asy" asy-temp-dir)) (if noext "" ".asy"))))

(defun asy-log-filename()
  (concat buffer-file-name ".log"))

(defun asy-compile()
  "Compile Asymptote code."
  (interactive)
  (if (and (boundp two-mode-bool) two-mode-bool)
      (lasy-compile)
    (progn
      (let*
          ((buffer-base-name (file-name-sans-extension (file-name-nondirectory buffer-file-name)))
           (asy-compile-command
            (if (eq asy-compilation-buffer 'never)
                (concat asy-command-location
                        asy-command
                        " "
                        buffer-base-name
                        ".asy 2>" (asy-log-filename))
              (concat  asy-command-location asy-command " -wait " buffer-base-name)
              ;;               (concat "{ empty=`" asy-command-location asy-command
              ;;                       " " buffer-base-name ".asy 3>&1 >&4 4>&-`; } 4>&1")
              )))
        (if (buffer-modified-p) (save-buffer))
        (asy-interneral-compile asy-compile-command)
        (when (eq asy-compilation-buffer 'never) (asy-error-message t))
        ))))

(defun asy-error-message(&optional P)
  (let ((asy-last-error
         (asy-log-field-string
          (asy-log-filename) 0)))
    (if (and asy-last-error (not (string= asy-last-error "")))
        (message (concat asy-last-error (if P "\nPress F4 to go to error" "")))
      )
    ))

(defun asy-log-field-string(Filename Field)
  "Return field of first line of file filename.
Fields are defined as 'field1: field2.field3:field4' . Field=0 <-> all fields"
  (with-temp-buffer
    (progn
      (insert-file Filename)
      (beginning-of-buffer)
      (if (re-search-forward "^\\(.*?\\): \\(.*?\\)\\.\\(.*?\\):\\(.*\\)$" (point-max) t)
          (match-string Field) nil))))

(defun asy-next-error(arg reset)
  (if (> emacs-major-version 21)
      (next-error arg reset)
    (next-error arg)))

(defun lasy-ask-visit-tem-compilation-buffer()
  "* Ask before visiting a temporary compilation buffer depending the value of `lasy-ask-about-temp-compilation-buffer'."
  (if lasy-ask-about-temp-compilation-buffer
      (y-or-n-p "Visit temporary buffer of compilation ? ") t)
  )

(defun asy-goto-error(&optional arg reset)
  "Go to point of last error within asy/lasy-mode."
  (interactive "P")
  (if (or (eq asy-compilation-buffer 'never)
          (and (boundp two-mode-bool) two-mode-bool))
      (let* ((log-file (asy-log-filename))
             (li_ (asy-log-field-string log-file 2))
             (co_ (asy-log-field-string log-file 3)))
        (if (and (boundp two-mode-bool) two-mode-bool) ;; Within Lasy-mode
            (if lasy-run-tex ; lasy-mode need the compilation of file.tex
                                        ; the error can be in Tex commands or in Asymptote comamds
                (progn
                  (if (or (eq asy-compilation-buffer 'never) lasy-compile-tex);; 
                      (if li_
                          (when (lasy-ask-visit-tem-compilation-buffer)
                            (let ((file (asy-log-field-string log-file 1)))
                              (find-file-other-window file)
                              (beginning-of-buffer)
                              (next-line (1- (string-to-number li_)))
                              (beginning-of-line)
                              (forward-char (1- (string-to-number co_)))))
                        (message "There is an error in your LaTeX code..."))
                    (when (lasy-ask-visit-tem-compilation-buffer) (asy-next-error arg reset))))
              (if li_
                  (progn
                    (let ((nline (- (1- (string-to-number li_)) lasy-error-ignore-number-line)))
                      (re-search-backward "\\\\begin{asy")
                      (if (> nline 0)
                          (progn
                            (next-line nline)
                            (beginning-of-line)
                            (forward-char (1- (string-to-number co_)))
                            (asy-error-message))
                        (progn
                          (beginning-of-buffer)
                          (re-search-forward "\\\\begin{asydef}" (point-max) t)
                          (next-line (string-to-number li_))
                          (beginning-of-line)
                          (forward-char (1- (string-to-number co_)))
                          (asy-error-message)
                          )
                        )))
                (message "No error.")))
          (if li_ ;;Pure asy-mode and compilation with shell-command
              (progn
                (goto-line (string-to-number li_))
                (forward-char (1- (string-to-number co_)))
                (asy-error-message))
            (progn (message "No error.")))))
    (asy-next-error arg reset)))

(defun asy-grep (Regexp)
  "Internal function used by asymptote."
  (let ((Strout "")
	(case-fold-search-asy case-fold-search))
    (progn
      (beginning-of-buffer)
      (setq case-fold-search nil)
      (while (re-search-forward Regexp (point-max) t)
        (setq Strout (concat Strout (match-string 0) "\n\n")))
      (setq case-fold-search case-fold-search-asy)
      (if (string= Strout "") "No match.\n" Strout))))


(defun asy-show-function-at-point()
  "Show the Asymptote definitions of the command at point."
  (interactive)
  (save-excursion
    (let ((cWord (current-word))
          (cWindow (selected-window)))
      (switch-to-buffer-other-window "*asy-help*")
      (if (> emacs-major-version 21)
          (call-process-shell-command
           (concat asy-command-location "asy -l") nil t nil)
        (insert (shell-command-to-string "asy -l")))
      (let ((rHelp (asy-grep (concat "^.*\\b" cWord "(\\(.\\)*?$"))))
        (erase-buffer)
        (insert rHelp))
      (asy-mode)
      (use-local-map nil)
      (goto-char (point-min))
      (select-window cWindow))))

(add-hook 'asy-mode-hook
	  (lambda ()
            ;; Make asy-mode work with other shells.
            (if (or running-xemacs-p (< emacs-major-version 22))
                (progn
                  (make-local-variable 'shell-file-name)
                  (set-variable 'shell-file-name "/bin/sh"))
              (set-variable 'shell-file-name "/bin/sh" t))
            (when (fboundp 'flyspell-mode) (flyspell-mode -1))
            (turn-on-font-lock)
            (column-number-mode t)
            ))

;;;###autoload (defun lasy-mode ())
;;; ************************************
;;; asy-mode mixed with LaTeX-mode: lasy
;;; ************************************
(if (locate-library "two-mode-mode")
    (progn
      (eval-after-load "two-mode-mode"
        '(progn ;; Redefine `two-mode-mode-update-mode' to use regexp.
           (defun two-mode-mode-update-mode ()
             (when (and two-mode-bool two-mode-update)
               (setq two-mode-update 0)
               (let ((mode-list second-modes)
                     (flag 0))
                 (while mode-list
                   (let ((mode (car mode-list))
                         (lm -1)
                         (rm -1))
                     (save-excursion 
                       (if (search-backward-regexp (cadr mode) nil t)
                           (setq lm (point))
                         (setq lm -1)))
                     (save-excursion
                       (if (search-backward-regexp (car (cddr mode)) nil t)
                           (setq rm (point))
                         (setq rm -1)))
                     (if (and (not (and (= lm -1) (= rm -1))) (>= lm rm))
                         (progn
                           (setq flag 1)
                           (setq mode-list '())
                           (two-mode-change-mode (car mode) (car (cdr (cddr mode)))))))
                   (setq mode-list (cdr mode-list)))
                 (if (= flag 0)
                     (two-mode-change-mode (car default-mode) (cadr default-mode))))))))
      (require 'two-mode-mode)
      (defun lasy-mode ()
        "Treat, in some cases, the current buffer as a literal Asymptote program."
        (interactive)
        (set (make-local-variable 'asy-insinuate-latex-p) asy-insinuate-latex-p)
        (setq default-mode    '("LaTeX" latex-mode)
              second-modes     '(("Asymptote"
                                  "\\\\begin{asy}.*$"
                                  "\\end{asy}"
                                  asy-mode)))
        (if two-mode-bool
            (progn
              (latex-mode)
              (asy-insinuate-latex)
              )
          (progn
            (two-mode-mode)))
        )
      
      (add-hook 'two-mode-switch-hook
                (lambda ()
                  (if (string-match "latex" (downcase mode-name))
                      (progn ;; Disable LaTeX-math-Mode within lasy-mode (because of incompatibility)
                        (when LaTeX-math-mode (LaTeX-math-mode))
                        (asy-insinuate-latex)))))
            
      ;; Solve a problem restoring a TeX file via desktop.el previously in lasy-mode.
      (if (boundp 'desktop-buffer-mode-handlers)
          (progn
            (defun asy-restore-desktop-buffer (desktop-b-f-name d-b-n d-b-m)
              (find-file desktop-b-f-name))
            (add-to-list 'desktop-buffer-mode-handlers
                         '(asy-mode . asy-restore-desktop-buffer))))
      )
  (progn
    (defvar two-mode-bool nil)))

(setq asy-latex-menu-item
      '(["Toggle lasy-mode"  lasy-mode :active (featurep 'two-mode-mode)]
        ["View asy picture near cursor"  lasy-compile :active t]"--"
        ("Compile OR View"
         ["PS"  lasy-view-ps :active t]
         ["PDF (pdflatex)" lasy-view-pdf-via-pdflatex :active t]
         ["PDF (ps2pdf)" lasy-view-pdf-via-ps2pdf :active t])
        ("Compile AND View"
         ["PS"  asy-master-tex-view-ps-f :active t]
         ["PDF (pdflatex)" asy-master-tex-view-pdflatex-f :active t]
         ["PDF (ps2pdf)" asy-master-tex-view-ps2pdf-f :active t])"--"
        ["Asymptote insinuates globally LaTeX"  asy-insinuate-latex-globally :active (not asy-insinuate-latex-globally-p)]
        ("Disable Asymptote insinuate Latex"
         ["locally"  asy-no-insinuate-locally :active t]
         ["globally"  asy-no-insinuate-globally :active t])
        ("Debugger Buffer"
         ["Visible" (setq asy-compilation-buffer 'visible) :style radio :selected (eq asy-compilation-buffer 'visible) :active t]
         ["Available" (setq asy-compilation-buffer 'available)  :style radio :selected (eq asy-compilation-buffer 'available) :active t]
         ["None" (setq asy-compilation-buffer 'none)  :style radio :selected (eq asy-compilation-buffer 'none) :active t]
         ["Never" (setq asy-compilation-buffer 'never)  :style radio :selected (eq asy-compilation-buffer 'never) :active t])
        ))
(if running-xemacs-p
    (setq asy-latex-menu-item (nconc '("Asymptote") asy-latex-menu-item))
  (setq asy-latex-menu-item (nconc '("Asymptote" :visible asy-insinuate-latex-p) asy-latex-menu-item)))


(eval-after-load "latex"
  '(progn
     (setq lasy-mode-map (copy-keymap LaTeX-mode-map))
     (setq LaTeX-mode-map-backup (copy-keymap LaTeX-mode-map))
     (when lasy-extra-key
       (define-key lasy-mode-map (kbd "<C-return>")
         (lambda ()
           (interactive)
           (lasy-view-ps nil nil t)))
       (define-key lasy-mode-map (kbd "<C-S-return>")
         (lambda ()
           (interactive)
           (lasy-view-ps t nil t)))
       (define-key lasy-mode-map (kbd "<M-return>")
         (lambda ()
           (interactive)
           (lasy-view-pdf-via-pdflatex nil nil t)))
       (define-key lasy-mode-map (kbd "<M-S-return>")
         (lambda ()
           (interactive)
           (lasy-view-pdf-via-pdflatex t nil t)))
       (define-key lasy-mode-map (kbd "<C-M-return>")
         (lambda ()
           (interactive)
           (lasy-view-pdf-via-ps2pdf nil nil t)))
       (define-key lasy-mode-map (kbd "<C-M-S-return>")
         (lambda ()
           (interactive)
           (lasy-view-pdf-via-ps2pdf t nil t)))
       (define-key lasy-mode-map  (kbd "<f4>") 'asy-goto-error))
     
     ;; Hack not totally safe.
     ;; Problems may occur if you customize the variables TeX-expand-list or TeX-command-list.
     ;; If you will never customize these variables, you can uncomment the following lines.
     ;;      (add-to-list 'TeX-expand-list
     ;;                   '("%a"
     ;;                     (lambda nil
     ;;                       asy-command-location)) t)
       
     ;;      (add-to-list 'TeX-command-list
     ;;                   '("asy-LaTeX" "%l \"%(mode)\\input{%t}\" && %aasy %s.asy && %l \"%(mode)\\input{%t}\" && %V"
     ;;                     TeX-run-command nil (latex-mode)
     ;;                     :help "Run LaTeX && Asymptote && LaTeX
     ;;              Be sure to have
     ;;              \\usepackage{graphicx}
     ;;              \\usepackage{asymptote}"))
       
     ;;      (add-to-list 'TeX-command-list
     ;;                   '("asy-pdflaTex" "pdflatex -shell-escape %t && %aasy %s.asy && pdflatex -shell-escape %t"
     ;;                     TeX-run-command nil (latex-mode)
     ;;                     :help "Run pdflatex && Asymptote && pdflatex
     ;;              Be sure to have
     ;;              \\usepackage{graphicx}
     ;;              \\usepackage{epstopdf}
     ;;              \\usepackage{asymptote}"))
       
     ;;      (add-to-list 'TeX-command-list
     ;;                   '("asy-ps" "%l \"%(mode)\\input{%t}\" && %aasy %s.asy && %(o?)dvips %d -o %f"
     ;;                     TeX-run-command nil (latex-mode)
     ;;                     :help "Run LaTeX && Asymptote && LaTeX
     ;;              Be sure to have
     ;;              \\usepackage{graphicx}
     ;;              \\usepackage{asymptote}"))
       
     ;;      (add-to-list 'TeX-command-list
     ;;                   '("asy-dvips-pdf" "%l \"%(mode)\\input{%t}\" && %aasy %s.asy && %(o?)dvips %d -o %f && ps2df14 -dPDFSETTINGS=/prepress -dAutoFilterGrayImages=false -dAutoFilterColorImages=false -dColorImageFilter=/FlateEncode -dGrayImageFilter=/FlateEncode -dAutoRotatePages=/None %f %s.pdf"
     ;;                     TeX-run-command nil (latex-mode)
     ;;                     :help "Run LaTeX && Asymptote && LaTeX && dvips && ps2pdf14
     ;;              Be sure to have
     ;;              \\usepackage{graphicx}
     ;;              \\usepackage{asymptote}"))
     
     (easy-menu-define asy-latex-mode-menu lasy-mode-map "Asymptote insinuates LaTeX" asy-latex-menu-item)
     ))

(defvar asy-insinuate-latex-p nil
  "Not nil when current buffer is insinuated by Asymptote.
May be a local variable.
For internal use.")

(defvar asy-insinuate-latex-globally-p nil
  "Not nil when all latex-mode buffers is insinuated by Asymptote.
For internal use.")


(defun asy-no-insinuate-locally ()
  (interactive)
  (set (make-local-variable 'asy-insinuate-latex-p) nil)
  (setq asy-insinuate-latex-globally-p nil)
  (if running-xemacs-p
      (easy-menu-remove-item nil nil "Asymptote")
    (menu-bar-update-buffers))
  (if (and (boundp 'two-mode-bool) two-mode-bool)
      (lasy-mode))
  (use-local-map LaTeX-mode-map-backup))


(defun asy-no-insinuate-globally ()
  (interactive)
  (if running-xemacs-p
      (easy-menu-remove-item nil nil "Asymptote")
    (easy-menu-remove-item LaTeX-mode-map nil "Asymptote"))
  (kill-local-variable asy-insinuate-latex-p)
  (setq-default asy-insinuate-latex-p nil)
  (setq asy-insinuate-latex-globally-p nil)
  (if (not running-xemacs-p)
      (menu-bar-update-buffers))
  (setq LaTeX-mode-map (copy-keymap LaTeX-mode-map-backup))
  ;;Disable lasy-mode in all latex-mode buffers.
  (when (featurep 'two-mode-mode)
    (mapc (lambda (buffer)
            (with-current-buffer buffer
              (when (and (buffer-file-name) (string= (file-name-extension (buffer-file-name)) "tex"))
                (latex-mode)
                (setq asy-insinuate-latex-p nil))))
          (buffer-list))))

;;;###autoload
(defun asy-insinuate-latex (&optional global)
  "Add a menu bar in current 'latex-mode' buffer and activate asy keys bindings.
If the optional parameter (only for internal use) 'global' is 't' then all the FUTURE 'latex-mode' buffers are insinuated.
To insinuate all (current and future) 'latex-mode' buffers, use 'asy-insinuate-latex-globally' instead.
You can automate this feature for all the 'latex-mode' buffers by inserting the five following lines in your .emacs initialization file:
   (eval-after-load \"latex\"
     '(progn
        ;; Add here your personal features for 'latex-mode':
        (asy-insinuate-latex t) ;; Asymptote insinuates globally Latex.
        ))"
  (interactive)
  (if (and (not asy-insinuate-latex-globally-p) (or global (string= major-mode "latex-mode")))
      (progn
        (if global
            (progn
              (setq asy-insinuate-latex-p t)
              (setq asy-insinuate-latex-globally-p t)
              (setq LaTeX-mode-map (copy-keymap lasy-mode-map))
              (if running-xemacs-p
                  (add-hook 'LaTeX-mode-hook
                            (lambda ()
                              (if asy-insinuate-latex-globally-p
                                  (easy-menu-add asy-latex-mode-menu lasy-mode-map))))))
          (progn
            (use-local-map lasy-mode-map)
            (easy-menu-add asy-latex-mode-menu lasy-mode-map)
            (set (make-local-variable 'asy-insinuate-latex-p) t)))        
        )))

(defun asy-insinuate-latex-globally ()
  "Insinuates all (current and future) 'latex-mode' buffers.
See 'asy-insinuate-latex'."
  (interactive)
  (asy-insinuate-latex t)
  (if running-xemacs-p
      (add-hook 'LaTeX-mode-hook
                (lambda ()
                  (if asy-insinuate-latex-globally-p
                      (easy-menu-add asy-latex-mode-menu lasy-mode-map)))))  
  (mapc (lambda (buffer)
          (with-current-buffer buffer
            (when (and
                   (buffer-file-name)
                   (string= (file-name-extension (buffer-file-name)) "tex"))
              (setq asy-insinuate-latex-p t)
              (use-local-map LaTeX-mode-map)
              (use-local-map lasy-mode-map)
              (easy-menu-add asy-latex-mode-menu lasy-mode-map))))
        (buffer-list)))

(defun lasy-inline-p()
  "Return nil if the option 'inline' is not used or if `lasy-compilation-inline-auto-detection' value is nil."
  (if lasy-compilation-inline-auto-detection
      (save-excursion
        (re-search-backward "^[^%]* *\\\\usepackage\\[ *inline *\\]{ *asymptote *}" 0 t))
    nil))

(defvar lasy-error-ignore-number-line 0)
(defvar lasy-run-tex nil)
(defun lasy-asydef()
  "Return the content between the tags \begin{asydef} and \end{asydef}.
Set the number of line into the variable `lasy-error-ignore-number-line'."
  (save-excursion
    (if (re-search-backward "\\\\begin{asydef}" 0 t)
        (let ((lb (line-number-at-pos)))
          (re-search-forward "\\\\begin{asydef} *\n\\(\\(\n\\|.\\)*\\)\\\\end{asydef}" (point-max))
          (setq lasy-error-ignore-number-line (- (- (line-number-at-pos) lb) 1))
          (when (and (< emacs-major-version 22) (not running-xemacs-p))
            (setq lasy-error-ignore-number-line (1- lasy-error-ignore-number-line)))
          (match-string 1))
      (progn
        (setq lasy-error-ignore-number-line 0)
        ""))))

(defun lasy-compile-tex()
  "Compile region between \\begin{asy}[text with backslash] and \\end{asy} passing by a reconstructed file .tex."
  (interactive)
  (setq lasy-run-tex t)
  (save-excursion
    (let* ((Filename (asy-get-temp-file-name t))
           (FilenameTex (concat Filename ".tex"))
           (asydef (lasy-asydef)))
      (save-excursion
        (beginning-of-buffer)
        (re-search-forward "\\(.\\|\n\\)*\\\\begin{document}.*\n")
        (write-region (concat (match-string 0) "\\begin{asydef}\n" asydef "\\end{asydef}\n") 0 FilenameTex))
      (re-search-forward "\\\\end{asy}")
      (re-search-backward "\\\\begin{asy}\\(\n\\|.\\)*?\\\\end{asy}")
      (write-region (match-string 0) 0 FilenameTex t)
      (with-temp-file FilenameTex
        (insert-file FilenameTex)
        (end-of-buffer)
        (insert "\n\\end{document}"))
      (let ((default-directory asy-temp-dir))
        (lasy-view-ps t Filename)))))

(defun lasy-compile()
  "Compile region between \\begin{asy} and \\end{asy}."
  (interactive)
  (if (or (lasy-inline-p) (progn ;; find \begin{asy}[any backslash]
                            (save-excursion
                              (re-search-forward "\\\\end{asy}" (point-max) t)
                              (re-search-backward "\\\\begin{asy}.*\\(\\[.*\\\\.*\\]\\)" 0 t))
                            (match-string 1)))
      (progn
        (lasy-compile-tex))
    (progn
      (setq lasy-run-tex nil)
      (save-excursion
        (let ((Filename (asy-get-temp-file-name))
              (asydef (lasy-asydef)))
          (re-search-forward "\\\\end{asy}")
          (re-search-backward "\\\\begin{asy}\\(\n\\|.\\)*?\\\\end{asy}")
          (write-region (match-string 0) 0 Filename)
          (with-temp-file Filename
            (insert-file Filename)
            (beginning-of-buffer)
            (if (re-search-forward "\\\\begin{asy}\\[\\(.*\\)\\]" (point-max) t)
                (progn
                  (replace-match (concat asydef "size(" (match-string 1) ");"))
                  (+ lasy-error-ignore-number-line 1))
              (when (re-search-forward "\\\\begin{asy}" (point-max) t)
                (replace-match asydef)))
            (while (re-search-forward "\\\\end{asy}" (point-max) t)
              (replace-match "")))
          (let*
              ((asy-compile-command
                (concat "( " asy-command-location
                        asy-command " -wait " Filename
                        " 2> " (asy-log-filename) " & ) >/dev/null")))
            (shell-command asy-compile-command)
            (run-at-time 1 nil (lambda ()(asy-error-message t)))))))))

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
        (error "You should never give the same name to the TeX file and the Asymptote file"))
    (save-excursion
      (end-of-buffer)
      (if (re-search-backward "asy-TeX-master-file\\(.\\)*$" 0 t)
          (replace-match (concat "asy-TeX-master-file: \"" asy-TeX-master-file "\""))
        (insert (concat "
/// Local Variables:
/// asy-TeX-master-file: \"" asy-TeX-master-file "\"
/// End:")) t))))

(defun asy-unset-master-tex ()
  "Set the local variable 'asy-TeX-master-file to 'nil.
This variable is used by 'asy-master-tex-view-ps"
  (interactive)
  (set (make-local-variable 'asy-TeX-master-file) nil)
  (save-excursion
    (end-of-buffer)
    (if (re-search-backward "^.*asy-TeX-master-file:.*\n" 0 t)
        (replace-match ""))))

(defun asy-master-tex-error ()
  "Asy-mode internal use..."
  (if (y-or-n-p "You try to compile the TeX document that contains this picture.
You must set the local variable asy-TeX-master-file.
Do you want set this variable now ?")
      (asy-set-master-tex) nil))

(defun asy-master-tex-view (Func-view &optional Force fromtex)
  "Compile the LaTeX document that contains the picture of the current Asymptote code with the function Func-view.
Func-view can be one of 'lasy-view-ps, 'lasy-view-pdf-via-pdflatex, 'lasy-view-pdf-via-ps2pdf."
  (interactive)
  (if (or
       (and (boundp two-mode-bool) two-mode-bool)
       (string= (downcase (substring mode-name 0 5)) "latex"))
      (progn ;; Current mode is lasy-mode or latex-mode not asy-mode
        (funcall Func-view Force nil fromtex))
    (if asy-TeX-master-file
        (if (string= asy-TeX-master-file
                     (file-name-sans-extension buffer-file-name))
            (error "You should never give the same name to the TeX file and the Asymptote file")
          (funcall Func-view  Force asy-TeX-master-file fromtex))
      (if (asy-master-tex-error)
          (funcall Func-view Force asy-TeX-master-file fromtex)))))


(defun asy-compilation-finish-function (buf msg)
  (when (and (eq asy-compilation-buffer 'none)
             (string-match "*asy-compilation*" (buffer-name buf)))
    (if (string-match "exited abnormally" msg)
        (message "Compilation errors, press C-x ` or  F4 to visit.")
      (progn
        (save-excursion
          (set-buffer buf)
          (beginning-of-buffer)
          (if (not (search-forward-regexp "[wW]arning" nil t))
              (progn
                ;;no errors/Warning, make the compilation window go away
                (run-at-time 0.5 nil (lambda (buf_)
                                       (delete-windows-on buf_)
                                       (kill-buffer buf_)) buf)
                (message "No compilation errors..."))
            (message "Compilation warnings...")))))))

(if  running-xemacs-p
    (setq compilation-finish-function 'asy-compilation-finish-function)
  (add-to-list 'compilation-finish-functions
               'asy-compilation-finish-function))

(defun asy-interneral-compile (command)
  (let* ((compilation-buffer-name-function
          (lambda (mj) "*asy-compilation*")))
    (cond
     ((eq asy-compilation-buffer 'available)
      (compile command)
      (delete-window (get-buffer-window "*asy-compilation*")))
     ((eq asy-compilation-buffer 'never)
      (write-region "" 0 (asy-log-filename) nil)
      (shell-command (concat "( " command " ) &> /dev/null"))))
    (when (and (not (eq asy-compilation-buffer 'available)) (not (eq asy-compilation-buffer 'never))) (compile  command))))

(defun lasy-view-ps (&optional Force  Filename fromtex)
  "Compile a LaTeX document embedding Asymptote code with latex->asy->latex->dvips and/or view the Poscript output.
If optional argument Force is t then force compilation."
  (interactive)
  (setq lasy-run-tex t)
  (setq lasy-compile-tex fromtex)
  (if (buffer-modified-p) (save-buffer))
  (let
      ((b-b-n  (if Filename Filename (file-name-sans-extension buffer-file-name)))
       (dirstderr (if (or (eq asy-compilation-buffer 'never) lasy-compile-tex)
                      (progn
                        (write-region "" 0 (asy-log-filename) nil)
                        (concat " 2>" (asy-log-filename))) "")))
    (if (or (file-newer-than-file-p
             (concat b-b-n ".tex")
             (concat b-b-n ".ps"))
            Force)
        (let
            ((asy-compile-command
              (concat  "latex -interaction=nonstopmode " b-b-n ".tex && "
                       "{ if [[ -f " b-b-n ".asy ]]; then " asy-command-location lasy-command " " b-b-n ".asy" dirstderr " && latex -interaction=nonstopmode " b-b-n ".tex  ; else true; fi;} && "
                       "dvips " b-b-n ".dvi -o " b-b-n ".ps && " ps-view-command " " b-b-n ".ps")))
          (asy-interneral-compile asy-compile-command))
      (start-process "" nil ps-view-command (concat  b-b-n ".ps")))))

(defun lasy-view-pdf-via-pdflatex (&optional Force Filename fromtex)
  "Compile a LaTeX document embedding Asymptote code with pdflatex->asy->pdflatex and/or view the PDF output.
If optional argument Force is t then force compilation.
Be sure to have
   \\usepackage{graphicx}
   \\usepackage{epstopdf}
   \\usepackage{asymptote}
in the preamble."
  (interactive)
  (setq lasy-run-tex t)
  (setq lasy-compile-tex fromtex)
  (if (buffer-modified-p) (save-buffer))
  (when (eq asy-compilation-buffer 'never) (write-region "" 0 (asy-log-filename) nil))
  (let
      ((b-b-n  (if Filename Filename (file-name-sans-extension buffer-file-name)))
       (dirstderr (if (or (eq asy-compilation-buffer 'never) lasy-compile-tex)
                      (progn
                        (write-region "" 0 (asy-log-filename) nil)
                        (concat " 2>" (asy-log-filename))) "")))
    (if (or (file-newer-than-file-p
             (concat b-b-n ".tex")
             (concat b-b-n ".pdf"))
            Force)
        (let
            ((asy-compile-command
              (concat  "pdflatex -shell-escape -interaction=nonstopmode " b-b-n ".tex && "
                       "{ if [[ -f " b-b-n ".asy ]]; then " asy-command-location lasy-command " " b-b-n ".asy" dirstderr " && pdflatex -shell-escape  -interaction=nonstopmode " b-b-n ".tex ; else true; fi; } && "
                       pdf-view-command " " b-b-n ".pdf")))
          (asy-interneral-compile asy-compile-command))
      (start-process "" nil pdf-view-command (concat b-b-n ".pdf")))))

(defun lasy-view-pdf-via-ps2pdf (&optional Force Filename fromtex)
  "Compile a LaTeX document embedding Asymptote code with latex->asy->latex->dvips->ps2pdf14 and/or view the PDF output.
If optional argument Force is t then force compilation."
  (interactive)
  (setq lasy-run-tex t)
  (setq lasy-compile-tex fromtex)
  (if (buffer-modified-p) (save-buffer))
  (when (eq asy-compilation-buffer 'never) (write-region "" 0 (asy-log-filename) nil))
  (let
      ((b-b-n  (if Filename Filename (file-name-sans-extension buffer-file-name)))
       (dirstderr (if (or (eq asy-compilation-buffer 'never) lasy-compile-tex)
                      (progn
                        (write-region "" 0 (asy-log-filename) nil)
                        (concat " 2>" (asy-log-filename))) "")))
    (if (or (file-newer-than-file-p
             (concat b-b-n ".tex")
             (concat b-b-n ".pdf"))
            Force)
        (let
            ((asy-compile-command
              (concat  "latex  -interaction=nonstopmode " b-b-n" .tex && "
                       "{ if [[ -f " b-b-n ".asy ]]; then " asy-command-location lasy-command " " b-b-n ".asy" dirstderr " && latex  -interaction=nonstopmode " b-b-n ".tex ; else true; fi;} && "
                       "dvips -q -Ppdf -ta4 " b-b-n ".dvi -o " b-b-n ".ps && "
                       "ps2pdf14 -dPDFSETTINGS=/prepress -dAutoFilterGrayImages=false -dAutoFilterColorImages=false -dColorImageFilter=/FlateEncode -dGrayImageFilter=/FlateEncode -dAutoRotatePages=/None " b-b-n ".ps " b-b-n ".pdf && "
                       pdf-view-command " " b-b-n ".pdf")))
          (asy-interneral-compile asy-compile-command))
      (start-process "" nil pdf-view-command (concat b-b-n ".pdf")))))

;; Goto to the forward/backward tempo's mark
;; (define-key asy-mode-map (kbd "<M-right>") 'tempo-forward-mark)
;; (define-key asy-mode-map (kbd "<M-left>")  'tempo-backward-mark)

;; Complete the tempo tag (the first three letters of a keyword)
(define-key asy-mode-map (kbd "<f3>") 'tempo-complete-tag)

;; Goto error of last compilation
(define-key asy-mode-map  (kbd "<f4>") 'asy-goto-error)

;; Save and compile the file with option -V
(define-key asy-mode-map  (kbd "C-c C-c") 'asy-compile)

;; Show the definitions of command at point
(define-key asy-mode-map  (kbd "C-c ?") 'asy-show-function-at-point)

;; new line and indent
(define-key asy-mode-map (kbd "RET") 'newline-and-indent)

(defun asy-master-tex-view-ps ()
  "Look at `asy-master-tex-view'"
  (interactive)
  (asy-master-tex-view 'lasy-view-ps nil t))
(define-key asy-mode-map (kbd "<C-return>") 'asy-master-tex-view-ps)

(defun asy-master-tex-view-ps-f ()
  "Look at `asy-master-tex-view'"
  (interactive)
  (asy-master-tex-view 'lasy-view-ps t t))
(define-key asy-mode-map (kbd "<C-S-return>") 'asy-master-tex-view-ps-f)

(defun asy-master-tex-view-pdflatex ()
  "Look at `asy-master-tex-view'"
  (interactive)
  (asy-master-tex-view 'lasy-view-pdf-via-pdflatex nil t))
(define-key asy-mode-map (kbd "<M-return>") 'asy-master-tex-view-pdflatex)

(defun asy-master-tex-view-pdflatex-f ()
  "Look at `asy-master-tex-view'"
  (interactive)
  (asy-master-tex-view 'lasy-view-pdf-via-pdflatex t t))
(define-key asy-mode-map (kbd "<M-S-return>") 'asy-master-tex-view-pdflatex-f)

(defun asy-master-tex-view-ps2pdf ()
  "Look at `asy-master-tex-view'"
  (interactive)
  (asy-master-tex-view 'lasy-view-pdf-via-ps2pdf nil t))
(define-key asy-mode-map (kbd "<C-M-return>") 'asy-master-tex-view-ps2pdf)

(defun asy-master-tex-view-ps2pdf-f ()
  "Look at `asy-master-tex-view'"
  (interactive)
  (asy-master-tex-view 'lasy-view-pdf-via-ps2pdf t t))
(define-key asy-mode-map (kbd "<C-M-S-return>") 'asy-master-tex-view-ps2pdf-f)

(provide `asy-mode)
