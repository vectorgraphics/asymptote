;;; asy-mode.el

;; Copyright (C) 2006-8
;; Author: Philippe IVALDI 20 August 2006
;; http://www.piprime.fr/
;; Modified by: John Bowman
;;
;; This program is free software ; you can redistribute it and/or modify
;; it under the terms of the GNU Lesser General Public License as published by
;; the Free Software Foundation ; either version 3 of the License, or
;; (at your option) any later version.
;;
;; This program is distributed in the hope that it will be useful, but
;; WITHOUT ANY WARRANTY ; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
;; Lesser General Public License for more details.
;;
;; You should have received a copy of the GNU Lesser General Public License
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

(defvar asy-mode-version "1.6")

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
In this hybrid mode the major mode is LaTeX when the cursor is in LaTeX code and becomes asy-mode when the cursor is between '\\begin{asy}' and '\\end{asy}'.
All the features of asy-mode are provided and the key binding C-c C-c of asy-mode compiles and views only the code of the picture where the cursor is.
Note that some keys binding are added to the LaTeX-mode-map in lasy-mode if the value of the variable lasy-extra-key is t (the default)
.
* C-return: compile (if the buffer/file is modified) and view the PostScript output with sequence [latex->[asy->latex]->dvips]->PSviewer
* M-return: same with pdf output and with the sequence [pdflatex->[asy->pdflatex]]->PDFviewer
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
This package has been tested in:
* Linux Debian Etch
- GNU Emacs 22.0.50.1
- GNU Emacs 21.4.1 (only basic errors management and basic font-lock features within lasy-mode are supported)
* WindowsXP
- GNU Emacs 22.0.990.1 (i386-mingw-nt5.1.2600)

This package seems to work with XEmacs 21.4 but not all the features are available (in particular syntax highlighting).

Report bugs to http://asymptote.sourceforge.net

Some variables can be customized: M-x customize-group <RET> asymptote <RET>."

  (setq c++-font-lock-extra-types (cons "true" c++-font-lock-extra-types)))

(require 'font-lock)
(require 'cc-mode)
(require 'cl) ;; Common Lisp extensions for Emacs
(require 'compile)
(require 'wid-edit)

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.asy$" . asy-mode))

(defvar running-xemacs-p (featurep 'xemacs))
(defvar running-unix-p (not (string-match "windows-nt\\|ms-dos" (symbol-name system-type))))

(when running-xemacs-p
  (defalias 'turn-on-font-lock-if-enabled 'ignore)
  (defalias 'line-number-at-pos 'line-number)
  (defvar temporary-file-directory (temp-directory))
  (defun replace-regexp-in-string (regexp rep string)
    (replace-in-string string regexp rep))
  )

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
  " 'visible means keep compilation buffer visible ;
 'available means keep compilation buffer available in other buffer but not visible;
 'none means delete compilation buffer automatically after a *successful* compilation.
 'never means don't open any window or buffer attached to the compilation process.
If the value is 'never':
* Emacs is suspended until the child program returns;
* the management of errors is poorer than with other value;
* the compilation doesn't modify your current window configuration."
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
this variable must end in /."
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
  :type 'string
  :group 'asymptote)

(defcustom lasy-latex-command "latex -halt-on-error"
  "* Command invoked to compile a .tex file with LaTeX."
  :type 'string
  :group 'asymptote)

(defcustom lasy-pdflatex-command "pdflatex -halt-on-error"
  "* Command invoked to compile a .tex file with pdflaTex."
  :type 'string
  :group 'asymptote)

(defcustom lasy-dvips-pre-pdf-command "dvips -Ppdf"
  "* Command invoked to convert a .dvi file to a temporary .ps file in order to
generate a final .pdf file."
  :type 'string
  :group 'asymptote)

(defcustom lasy-dvips-command "dvips -q"
  "* Command invoked to convert a .dvi file to a final .ps file."
  :type 'string
  :group 'asymptote)

(defcustom lasy-ps2pdf-command "ps2pdf14"
  "* Command invoked to convert a .dvi file to .ps file."
  :type 'string
  :group 'asymptote)

(defcustom asy-temp-dir temporary-file-directory
  "*The name of a directory for Asy's temporary files.
Such files are generated by functions like
`asy-compile' when lasy-mode is enable."
  :type 'directory
  :group 'asymptote)

(defcustom ps-view-command (if running-unix-p "gv" "")
  "Command to view a PostScript file generated by compiling a tex file within lasy-mode.
This variable is not used when running the Windows OS.
See `asy-open-file'."
  :type 'string
  :group 'asymptote)

(defcustom pdf-view-command
  (if running-unix-p
      "xpdf" "")
  "Command to view a pdf file generated by compiling a tex file within lasy-mode.
This variable is not used when running the Windows OS.
See `asy-open-file'."
  :type 'string
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
  (if (locate-library "asy-keywords.el")
      (load "asy-keywords.el")
    (progn
      ;; Use dummy keyword definitions if asy-keywords.el is not found:
      (defvar asy-keyword-name nil)
      (defvar asy-type-name nil)
      (defvar asy-function-name nil)
      (defvar asy-variable-name nil)))

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

  (defface asy-link-face ;; widget-field-face
    `((t
       (:underline t)))
    "Face used to highlighting the links."
    :group 'asymptote)

  (font-lock-add-keywords
   'asy-mode
   '(("\\[.*?\\.asy\\]" . 'asy-link-face)))
  )

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

(defun asy-protect-file-name(Filename)
  (concat "\"" Filename "\""))

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
      (lasy-compile) ;; compile asy code in a TeX file.
    (progn           ;; compile asy code in a asy file.
      (let*
          ((buffer-base-name (file-name-sans-extension (file-name-nondirectory buffer-file-name)))
           (asy-compile-command
            (concat  asy-command-location asy-command
                     (if (eq asy-compilation-buffer 'never)
                         " " " -wait ")
                     (asy-protect-file-name buffer-base-name))))
        (if (buffer-modified-p) (save-buffer))
        (message "%s" asy-compile-command)
        (asy-internal-compile asy-compile-command t t)))))

(defun asy-error-message(&optional P)
  (let ((asy-last-error
         (asy-log-field-string
          (asy-log-filename) 0)))
    (if (and asy-last-error (not (string= asy-last-error "")))
        (message (concat asy-last-error (if P "\nPress F4 to go to error" "")))
      (when (and (boundp two-mode-bool) two-mode-bool lasy-run-tex (not (zerop asy-last-compilation-code)))
        (message "The LaTeX code may be incorrect.")))))

(defun asy-log-field-string(Filename Field)
  "Return field of first line of file filename.
Fields are defined as 'field1: field2.field3:field4' . Field=0 <-> all fields"
  (let ((view-inhibit-help-message t))
    (with-temp-buffer
      (progn
        (insert-file Filename)
        (beginning-of-buffer)
        (if (re-search-forward "^\\(.*?\\): \\(.*?\\)\\.\\(.*?\\):\\(.*\\)$" (point-max) t)
            (match-string Field) nil)))))

(defun asy-next-error(arg reset)
  (if (> emacs-major-version 21)
      (next-error arg reset)
    (next-error arg)))

(defun lasy-ask-visit-tem-compilation-buffer()
  "* Ask before visiting a temporary compilation buffer depending the value of `lasy-ask-about-temp-compilation-buffer'."
  (if lasy-ask-about-temp-compilation-buffer
      (y-or-n-p "Visit temporary buffer of compilation ? ") t))

(defun lasy-place-cursor-to-error(Filename li co)
  (save-excursion
    (with-temp-buffer
      (insert-file-contents
       (if running-unix-p Filename
         (replace-regexp-in-string
          "//" ":/"
          (replace-regexp-in-string "/cygdrive/" "" Filename)))) ;; Not right,
;;;maybe take a look at the code of compilation-find-file
      (beginning-of-buffer)
      (next-line (1- (string-to-number li)))
      (setq line-err
            (buffer-substring-no-properties
             (progn (beginning-of-line) (point))
             (progn (end-of-line) (point))))))
  (beginning-of-buffer)
  (search-forward line-err)
  (beginning-of-line)
  (forward-char (1- (string-to-number co))))

(defun asy-goto-error(&optional arg reset)
  "Go to point of last error within asy/lasy-mode."
  (interactive "P")
  (if (or (eq asy-compilation-buffer 'never)
          (and (boundp two-mode-bool) two-mode-bool))
      (let* ((log-file (asy-log-filename))
             (li_ (asy-log-field-string log-file 2))
             (co_ (asy-log-field-string log-file 3)))
        (if (and (boundp two-mode-bool) two-mode-bool) ;; Within Lasy-mode
            (progn ;; lasy-mode need the compilation of file.tex
              ;; the error can be in Tex commands or in Asymptote commands
              (if (eq asy-compilation-buffer 'never) ;; Find error in the log file.
                  (if li_ ;; Asy error found in the log-file
                      (progn
                        (lasy-place-cursor-to-error
                         (asy-log-field-string log-file 1) li_ co_)
                        (asy-error-message))
                    (message "There is an error in your LaTeX code..."))
                (if (or running-xemacs-p (< emacs-major-version 22))
                    (when (lasy-ask-visit-tem-compilation-buffer)
                      (next-error arg))
                  (let ((msg)) ;; Find error in the compilation buffer
                    (save-excursion
                      (set-buffer (next-error-find-buffer))
                      (when reset
                        (setq compilation-current-error nil))
                      (let* ((columns compilation-error-screen-columns)
                             (last 1)
                             (loc (compilation-next-error (or arg 1) nil
                                                          (or compilation-current-error
                                                              compilation-messages-start
                                                              (point-min))))
                             (end-loc (nth 2 loc))
                             (marker (point-marker)))
                        (setq compilation-current-error (point-marker)
                              overlay-arrow-position
                              (if (bolp)
                                  compilation-current-error
                                (copy-marker (line-beginning-position)))
                              loc (car loc)))
                      (if (re-search-forward "^\\(.*?\\): \\(.*?\\)\\.\\(.*?\\):\\(.*\\)$" (point-max) t)
                          (progn
                            (setq msg (match-string 0)
                                  log-file (match-string 1)
                                  li_ (match-string 2)
                                  co_ (match-string 3)))
                        (error "Not other errors.")))
                    (lasy-place-cursor-to-error log-file li_ co_)
                    (message msg)))))
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

(defun asy-widget-open-file-at-pos (widget &optional event)
  ""
  (kill-buffer (current-buffer))
  (find-file (widget-get widget :follow-link))
  (goto-line (string-to-number (widget-get widget :value))))

(defun asy-show-function-at-point()
  "Show the Asymptote definitions of the command at point."
  (interactive)
  (save-excursion
    (let ((cWord (current-word))
          (cWindow (selected-window)))
      (switch-to-buffer-other-window "*asy-help*")
      (fundamental-mode)
      (setq default-directory "/")
      (if (> emacs-major-version 21)
          (call-process-shell-command
           (concat asy-command-location "asy -l --where") nil t nil)
        (insert (shell-command-to-string "asy -l --where")))
      (let ((rHelp (asy-grep (concat "^.*\\b" cWord "(\\(.\\)*?$")))
            (tag)(file)(line))
        (erase-buffer)
        (insert rHelp)
        (beginning-of-buffer)
        (while (re-search-forward "\\(.*\\): \\([0-9]*\\)\\.\\([0-9]*\\)" (point-max) t)
          (setq file (match-string 1)
                line (match-string 2)
                tag (file-name-nondirectory file))
          (widget-create `(file-link
                           :tag ,tag
                           :follow-link ,file
                           :value ,line
                           :action asy-widget-open-file-at-pos
                           ))))
      (beginning-of-buffer)
      (while (re-search-forward "\\(.*: [0-9]*\\.[0-9]*\\)" (point-max) t)
        (replace-match ""))
      (asy-mode)
      (use-local-map widget-keymap)
      (widget-setup)
      (goto-char (point-min))
      (select-window cWindow))))

(add-hook 'asy-mode-hook
	  (lambda ()
	    (c-set-style "gnu");
	    (c-set-offset (quote topmost-intro-cont) 0 nil)
	    (make-local-variable 'c-label-minimum-indentation)
	    (setq c-label-minimum-indentation 0)
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

      (defvar lasy-fontify-asy-p nil
        "Variable to communicate with `font-lock-unfontify-region'.
Internal use, don't set in any fashion.")
      (setq lasy-fontify-asy-p nil)

      (eval-after-load "two-mode-mode"
        '(progn
           ;; Redefine `two-mode-mode-update-mode' to use regexp.
           (defun two-mode-mode-update-mode ()
             "Redefined in `asy-mode.el' to use regexp"
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
                     (two-mode-change-mode (car default-mode) (cadr default-mode))))))

           (defun two-mode-change-mode (to-mode func)
             "Redefined in asy-mode.
Change the variable `lasy-fontify-asy-p' according to the value of func and
the current mode."
             (if (string= to-mode mode-name)
                 t
               (progn
                 (setq lasy-fontify-asy-p (eq func 'asy-mode))
                 (funcall func)
                 (hack-local-variables)
                 (two-mode-mode-setup)
                 (if two-mode-switch-hook
                     (run-hooks 'two-mode-switch-hook))
                 (if (eq font-lock-mode t)
                     (font-lock-fontify-buffer))
                 (turn-on-font-lock-if-enabled))))
           ))


      (require 'two-mode-mode)

      (defun lasy-mode ()
        "Treat, in some cases, the current buffer as a literal Asymptote program."
        (interactive)
        (save-excursion
          (let ((prefix
                 (progn
                   (goto-char (point-max))
                   (re-search-backward "^\\([^\n]+\\)Local Variables:"
                                       (- (point-max) 3000) t)
                   (match-string 1)))
                (pos-b (point)))
            (when
                (and prefix
                     (progn
                       (re-search-forward (regexp-quote
                                           (concat prefix
                                                   "End:")) (point-max) t)
                       (re-search-backward (concat "\\(" prefix "mode: .*\\)") pos-b t))
                     )
              (error (concat "lasy-mode can not work if a mode is specified as local file variable.
You should remove the line " (int-to-string (line-number-at-pos)))))))
        (set (make-local-variable 'asy-insinuate-latex-p) asy-insinuate-latex-p)
        (make-local-variable 'lasy-fontify-asy-p)
        (when (< emacs-major-version 22)
          (make-local-variable 'font-lock-keywords-only))
        (setq default-mode    '("LaTeX" latex-mode)
              second-modes     '(("Asymptote"
                                  "^\\\\begin{asy}.*$"
                                  "^\\\\end{asy}"
                                  asy-mode)))
        (if two-mode-bool
            (progn
              (latex-mode)
              (asy-insinuate-latex))
          (progn
            (two-mode-mode)
            )))

      (when (not running-xemacs-p)
        (defadvice TeX-command-master (around asy-choose-compile act)
          "Hack to circumvent the preempt of 'C-c C-c' by AucTeX within `lasy-mode'."
          (if (string-match "asymptote" (downcase mode-name))
              (asy-compile)
            ad-do-it)))

      (add-hook 'two-mode-switch-hook
                (lambda ()
                  (if (eq major-mode 'latex-mode)
                      (progn ;; Switch to latex-mode
                        ;; Disable LaTeX-math-Mode within lasy-mode (because of incompatibility)
                        (when LaTeX-math-mode (LaTeX-math-mode -1))
                        (asy-insinuate-latex)
                        (when (< emacs-major-version 22)
                          (setq font-lock-keywords-only nil)))
                    (progn ;; Switch to asy-mode
                      (when (< emacs-major-version 22)
                        (setq font-lock-keywords-only t))
                      ))))
      ;; (setq two-mode-switch-hook nil)

      ;; Solve a problem restoring a TeX file via desktop.el previously in lasy-mode.
      (if (boundp 'desktop-buffer-mode-handlers)
          (progn
            (defun asy-restore-desktop-buffer (desktop-b-f-name d-b-n d-b-m)
              (find-file desktop-b-f-name))
            (add-to-list 'desktop-buffer-mode-handlers
                         '(asy-mode . asy-restore-desktop-buffer))))

      ;; Functions and 'advises' to restrict 'font-lock-unfontify-region'
      ;; and 'font-lock-fontify-syntactically-region' within lasy-mode
      ;; Special thanks to Olivier Ramaré for his help.
      (when (and (fboundp 'font-lock-add-keywords) (> emacs-major-version 21))
        (defun lasy-mode-at-pos (pos &optional interior strictly)
          "If point at POS is in an asy environment return the list (start end)."
          (save-excursion
            (save-match-data
              (goto-char pos)
              (let* ((basy
                      (progn
                        (unless strictly (end-of-line))
                        (when (re-search-backward "^\\\\begin{asy}" (point-min) t)
                          (when interior (next-line))
                          (point))))
                     (easy
                      (and basy
                           (progn
                             (when (re-search-forward "^\\\\end{asy}" (point-max) t)
                               (when interior (previous-line)(beginning-of-line))
                               (point))))))
                (and basy easy
                     (> pos (- basy (if interior 12 0)))
                     (< pos (+ easy (if interior 10 0)))
                     (list basy easy))))))

        (defun lasy-region (start end &optional interior)
          "If the region 'start to end' contains the beginning or
the end of an asy environment return the list of points where
the asy environment starts and ends."
          (let* ((beg (min start end))
                 (lim (max start end)))
            (or (lasy-mode-at-pos beg interior)
                (save-match-data
                  (save-excursion
                    (goto-char beg)
                    (and (re-search-forward "^\\\\begin{asy}" lim t)
                         (lasy-mode-at-pos (point) interior)))))))

        (defun lasy-tags (start end)
          "Return associated list of points where the tags starts and ends
restricted to the region (start end).
\"b\" associated with (start-beginTag end-beginTag),
\"e\" associated with (start-endTag end-endTag)."
          (let*
              ((beg (min start end))
               (lim (max start end))
               out)
            (save-excursion
              (goto-char beg)(beginning-of-line)
              (while
                  (when (re-search-forward "^\\\\begin{asy}.*" lim t)
                    (push (list
                           (progn (beginning-of-line)(point))
                           (progn (end-of-line)(point))) out)))
              (goto-char beg)(beginning-of-line)
              (while
                  (when (re-search-forward "^\\\\end{asy}" lim t)
                    (push (list
                           (progn (beginning-of-line)(point))
                           (progn (end-of-line)(point))) out)))
              out)))

        (defun lasy-restrict-region (start end &optional interior)
          "If the region 'start to end' contains the beginning or
the end of an asy environment, returns the list of points wich
restricts the region to the asy environment.
Else, return (start end)."
          (let*
              ((beg (min start end))
               (lim (max start end))
               (be (if (lasy-mode-at-pos beg)
                       beg
                     (or (save-excursion
                           (goto-char beg)
                           (when (re-search-forward "^\\\\begin{asy}.*" lim t)
                             (unless interior (beginning-of-line))
                             (point)))
                         beg)))
               (en (or (save-excursion
                         (goto-char be)
                         (when (re-search-forward "^\\\\end{asy}" lim t)
                           (when interior (beginning-of-line))
                           (point)))
                       lim)))
            (list be en)))

        (defun lasy-parse-region (start end)
          "Return a list ((a (start1 end1)) (b (start2 end2)) [...]).
where a, b, ... are nil or t; t means the region from 'startX' through 'endX' (are points)
is in a asy environnement."
          (let (regasy out rr brr err tags)
            (save-excursion
              (goto-char start)
              (while (< (point) end)
                (setq regasy (lasy-region (point) end))
                (if regasy
                    (progn
                      (setq rr (lasy-mode-at-pos (point)))
                      (setq brr (and rr (nth 0 rr))
                            err (and rr (nth 1 rr)))
                      (if rr
                          (progn
                            (push (list t (list (max 1 (1- (point))) (min end err))) out)
                            (goto-char (min end err)))
                        (progn
                          (push (list nil (list (point) (nth 0 regasy))) out)
                          (goto-char (1+ (nth 0 regasy))))))
                  (progn
                    (push (list nil (list (min (1+ (point)) end) end)) out)
                    (goto-char end)))
                ))
            ;; Put start and end of tag in latex fontification.
            (setq tags (lasy-tags start end))
            (dolist (tag tags) (push (list nil tag) out))
            (reverse out)))

        (defadvice font-lock-unfontify-region
          (around asy-font-lock-unfontify-region (beg end))
          (if two-mode-bool
              (let ((rstate (lasy-parse-region beg end))
                    curr reg asy-fontify latex-fontify)
                (while (setq curr (pop rstate))
                  (setq reg (nth 1 curr))
                  (setq asy-fontify (and (nth 0 curr) lasy-fontify-asy-p)
                        latex-fontify (and (not (nth 0 curr))
                                           (not lasy-fontify-asy-p)))
                  (when (or asy-fontify latex-fontify)
                    (setq beg (nth 0 reg)
                          end (nth 1 reg))
                    (save-excursion
                      (save-restriction
                        (narrow-to-region beg end)
                        ad-do-it
                        (widen))))))
            ad-do-it))

        (ad-activate 'font-lock-unfontify-region)
        ;; (ad-deactivate 'font-lock-unfontify-region)

        (defadvice font-lock-fontify-syntactically-region
          (around asy-font-lock-fontify-syntactically-region
                  (start end &optional loudly))
          (if (and two-mode-bool (eq major-mode 'asy-mode))
              (let*((reg (lasy-restrict-region start end)))
                (save-restriction
                  (setq start (nth 0 reg) end (nth 1 reg))
                  (narrow-to-region start end)
                  (condition-case nil
                      ad-do-it
                    (error nil))
                  (widen)
                  ))
            ad-do-it))

        (ad-activate 'font-lock-fontify-syntactically-region)
        ;; (ad-deactivate 'font-lock-fontify-syntactically-region)

        (defadvice font-lock-default-fontify-region
          (around asy-font-lock-default-fontify-region
                  (beg end loudly))
          (if two-mode-bool
              (let ((rstate (lasy-parse-region beg end))
                    asy-fontify latex-fontify curr reg)
                (while (setq curr (pop rstate))
                  (setq reg (nth 1 curr))
                  (setq asy-fontify (and (nth 0 curr) lasy-fontify-asy-p)
                        latex-fontify (and (not (nth 0 curr))
                                           (not lasy-fontify-asy-p)))
                  (when (or asy-fontify latex-fontify)
                    (setq beg (nth 0 reg)
                          end (nth 1 reg))
                    (save-excursion
                      (save-restriction
                        (narrow-to-region beg end)
                        (condition-case nil
                            ad-do-it
                          (error nil))
                        (widen)
                        )))))
            ad-do-it))

        (ad-activate 'font-lock-default-fontify-region)
        ;; (ad-deactivate 'font-lock-default-fontify-region)

        ))
  (progn
    (defvar two-mode-bool nil)
    (defun lasy-mode ()
      (message "You must install the package two-mode-mode.el."))))

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

(defun asy-insinuate-latex-maybe ()
  "This function is added to `LaTeX-mode-hook' to define the environment 'asy'
and, eventually, set its indentation.
For internal use only."
  (when (or asy-insinuate-latex-globally-p
            (save-excursion
              (beginning-of-buffer)
              (save-match-data
                (search-forward "\\begin{asy}" nil t))))
    (asy-insinuate-latex))
  (LaTeX-add-environments
   '("asy"  (lambda (env &rest ignore)
              (unless asy-insinuate-latex-p (asy-insinuate-latex))
              (LaTeX-insert-environment env)))))

;; (add-hook 'after-init-hook
;;           (lambda ()
(eval-after-load "latex"
  '(progn
     (add-hook 'LaTeX-mode-hook 'asy-insinuate-latex-maybe)
     (setq lasy-mode-map (copy-keymap LaTeX-mode-map))
     (setq LaTeX-mode-map-backup (copy-keymap LaTeX-mode-map))

     (defadvice TeX-add-local-master (after asy-adjust-local-variable ())
       "Delete the line that defines the mode in a file .tex because two-mode-mode reread
the local variables after switching mode."
       (when (string= (file-name-extension buffer-file-name) "tex")
         (save-excursion
           (goto-char (point-max))
           (delete-matching-lines
            "mode: latex"
            (re-search-backward "^\\([^\n]+\\)Local Variables:"
                                (- (point-max) 3000) t)
            (re-search-forward (regexp-quote
                                (concat (match-string 1)
                                        "End:"))) nil))))
     (ad-activate 'TeX-add-local-master)
     ;; (ad-deactivate 'TeX-add-local-master)

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

     (easy-menu-define asy-latex-mode-menu lasy-mode-map "Asymptote insinuates LaTeX" asy-latex-menu-item)
     ))
;; ))

(defvar asy-insinuate-latex-p nil
  "Not nil when current buffer is insinuated by Asymptote.
May be a local variable.
For internal use.")

(defvar asy-insinuate-latex-globally-p nil
  "Not nil when all latex-mode buffers is insinuated by Asymptote.
For internal use.")

(defun asy-set-latex-asy-indentation ()
  "Set the indentation of environnment 'asy' like the environnment 'verbatim' is."
  ;; Regexp matching environments with indentation at col 0 for begin/end.
  (set (make-local-variable 'LaTeX-verbatim-regexp)
       (concat (default-value 'LaTeX-verbatim-regexp) "\\|asy"))
  ;; Alist of environments with special indentation.
  (make-local-variable 'LaTeX-indent-environment-list)
  (add-to-list 'LaTeX-indent-environment-list
               '("asy" current-indentation)))

(defun asy-unset-latex-asy-indentation ()
  "Unset the indentation of environnment 'asy' like the environnment 'verbatim' is."
  (set (make-local-variable 'LaTeX-verbatim-regexp)
       (default-value 'LaTeX-verbatim-regexp))
  (set (make-local-variable 'LaTeX-indent-environment-list)
       (default-value 'LaTeX-indent-environment-list)))

(defun asy-no-insinuate-locally ()
  (interactive)
  (set (make-local-variable 'asy-insinuate-latex-p) nil)
  (setq asy-insinuate-latex-globally-p nil)
  (asy-unset-latex-asy-indentation)
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
                (asy-unset-latex-asy-indentation)
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
        (asy-set-latex-asy-indentation)
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
See `asy-insinuate-latex'."
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
              (asy-set-latex-asy-indentation)
              (easy-menu-add asy-latex-mode-menu lasy-mode-map))))
        (buffer-list)))

(defun lasy-inline-p()
  "Return nil if the option 'inline' is not used or if `lasy-compilation-inline-auto-detection' value is nil."
  (if lasy-compilation-inline-auto-detection
      (save-excursion
        (re-search-backward "^[^%]* *\\\\usepackage\\[ *inline *\\]{ *asymptote *}" 0 t))
    nil))

(defvar lasy-run-tex nil)
(defun lasy-asydef()
  "Return the content between the tags \begin{asydef} and \end{asydef}."
  (save-excursion
    (if (re-search-backward "\\\\begin{asydef}" 0 t)
        (buffer-substring
         (progn (next-line)(beginning-of-line)(point))
         (progn (re-search-forward "\\\\end{asydef}")
                (previous-line)(end-of-line)
                (point)))
      "")))

(defun lasy-compile-tex()
  "Compile region between \\begin{asy}[text with backslash] and \\end{asy} through a reconstructed file .tex."
  (interactive)
  (setq lasy-run-tex t)
  (save-excursion
    (let* ((Filename (asy-get-temp-file-name t))
           (FilenameTex (concat Filename ".tex"))
           (asydef (lasy-asydef)))
      (save-excursion
        (beginning-of-buffer)
        (write-region (point)
                      (progn
                        (re-search-forward "\\\\begin{document}.*\n")
                        (point)) FilenameTex)
        (write-region (concat "\\begin{asydef}\n" asydef "\n\\end{asydef}\n") 0 FilenameTex t))
      (re-search-backward "\\\\begin{asy}")
      (write-region (point) (progn
                              (re-search-forward "\\\\end{asy}")
                              (point)) FilenameTex t)
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
        (lasy-compile-tex)) ;; a temporary TeX file must be reconstructed.
    (progn
      (setq lasy-run-tex nil)
      (save-excursion
        (let ((Filename (asy-get-temp-file-name))
              (asydef (lasy-asydef)))
          (write-region (match-string 0) 0 Filename)
          (re-search-backward "\\\\begin{asy}")
          (write-region (point) (progn
                                  (re-search-forward "\\\\end{asy}")
                                  (point)) Filename)
          (with-temp-file Filename
            (insert-file-contents Filename)
            (beginning-of-buffer)
            (if (re-search-forward "\\\\begin{asy}\\[\\(.*\\)\\]" (point-max) t)
                (let ((sz (match-string 1)))
                  (replace-match "")
                  (insert (concat asydef "\nsize(" sz ");")))
              (when (re-search-forward "\\\\begin{asy}" (point-max) t)
                (replace-match "")
                (insert asydef)))
            (while (re-search-forward "\\\\end{asy}" (point-max) t)
              (replace-match "")))
          (let* ((asy-compile-command
                  (concat  asy-command-location
                           asy-command
                           (if (eq asy-compilation-buffer 'never)
                               " " " -wait ")
                           (asy-protect-file-name Filename))))
            (asy-internal-compile
             asy-compile-command t
             (not (eq asy-compilation-buffer 'never)))))))))

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
       (string-match "latex" (downcase  mode-name)))
      (progn ;; Current mode is lasy-mode or latex-mode not asy-mode
        (funcall Func-view Force nil fromtex))
    (if asy-TeX-master-file
        (if (string= asy-TeX-master-file
                     (file-name-sans-extension buffer-file-name))
            (error "You should never give the same name to the TeX file and the Asymptote file")
          (funcall Func-view  Force asy-TeX-master-file fromtex))
      (if (asy-master-tex-error)
          (funcall Func-view Force asy-TeX-master-file fromtex)))))

(defvar asy-last-compilation-code nil
  "Code returned by the last compilation with `compile'.")

(defvar asy-compilation-auto-close nil
  "Variable to communicate with `asy-compilation-finish-function'.
Do not set this variable in any fashion.")

(defun asy-compilation-finish-function (buf msg)
  "Function to automatically close the compilation buffer '*asy-compilation*'
when no error or warning occurs."
  (when (string-match "*asy-compilation*" (buffer-name buf))
    (when (and asy-compilation-auto-close
               (eq asy-compilation-buffer 'none))
      (setq asy-compilation-auto-close nil)
      (if (not (string-match "exited abnormally" msg))
          (progn
            (save-excursion
              (set-buffer buf)
              (beginning-of-buffer)
              (if (not (search-forward-regexp "[wW]arning" nil t))
                  (when (not (eq asy-compilation-buffer 'visible))
                    ;;no errors/Warning, make the compilation window go away
                    (run-at-time 0.5 nil (lambda (buf_)
                                           (delete-windows-on buf_)
                                           (kill-buffer buf_)) buf)
                    (message (replace-regexp-in-string "\n" "" msg)))
                (message "Compilation warnings..."))))))))

(if  running-xemacs-p
    (setq compilation-finish-function 'asy-compilation-finish-function)
  (add-to-list 'compilation-finish-functions
               'asy-compilation-finish-function))

(defun asy-compilation-wait(&optional pass auto-close)
  "Wait for process in *asy-compilation* exits.
If pass is 't' don't wait.
If auto-close is 't' close the window if the process exit with success."
  (setq asy-compilation-auto-close auto-close)
  (let* ((buff (get-buffer "*asy-compilation*"))
         (comp-proc (get-buffer-process buff)))
    (while (and comp-proc
                (not (eq (process-status comp-proc) 'exit))
                (not pass))
      (setq comp-proc (get-buffer-process buff))
      (sit-for 1)
      (message "Waiting process...") ;; need message in Windows system
      )
    (message "") ;; Erase previous message.
    (if (and (not pass) comp-proc)
        (setq asy-last-compilation-code (process-exit-status comp-proc))
      (setq asy-last-compilation-code 0))
    (when (and (eq asy-compilation-buffer 'available)
               (zerop asy-last-compilation-code))
      (delete-windows-on buff))))


(defun asy-internal-shell (command &optional pass)
  "Execute 'command' in a inferior shell discarding output and
redirecting stderr in the file given by the command `asy-log-filename'.
`asy-internal-shell' waits for PROGRAM to terminate and returns a numeric exit status.
The variable `asy-last-compilation-code' is always set to the exit status.
The optional argument pass, for compatibility, is not used."
  (let* ((log-file (asy-log-filename))
         (discard (if pass 0 nil))
         (status
          (progn
            (let ((view-inhibit-help-message t))(write-region "" 0 log-file nil))
            (message "%s" command)
            (call-process shell-file-name nil (list nil log-file) nil shell-command-switch command))))
    (setq asy-last-compilation-code (if status status 0))
    (if status status nil)))

;; (defun asy-internal-shell (command &optional pass)
;;   "Execute 'command' in a inferior shell discarding output and
;; redirecting stderr in the file given by the command `asy-log-filename'.
;; pass non-nil means `asy-internal-shell' returns immediately with nil value.
;; Otherwise it waits for PROGRAM to terminate and returns a numeric exit status.
;; The variable `asy-last-compilation-code' is always set to the exit status or 0 if the
;; process returns immediately."
;;   (let* ((log-file (asy-log-filename))
;;         (discard (if pass 0 nil))
;;         (status
;;          (progn
;;            (let ((inhibit-redisplay t))(write-region "" 0 log-file nil))
;;            (message "%s" command)
;;            (call-process shell-file-name nil (list discard log-file) nil shell-command-switch command))))
;;     (setq asy-last-compilation-code (if status status 0))
;;     (when pass (sit-for 1))
;;     (if status status nil)))

(defun asy-internal-compile (command &optional pass auto-close stderr)
  "Execute command.
pass non-nil means don't wait the end of the process.
auto-close non-nil means automatically close the compilation buffer.
stderr non-nil means redirect the standard output error to the file
returned by `asy-log-filename'.
In this case command is running in an inferior shell without any output and
the parameter auto-close is not used (see `asy-internal-shell')."
  (setq asy-last-compilation-code -1)
  (let* ((compilation-buffer-name "*asy-compilation*")
         (compilation-buffer-name-function (lambda (mj) compilation-buffer-name)))
    (if (or stderr (eq asy-compilation-buffer 'never))
        (progn
          (asy-internal-shell command pass)
          (asy-error-message t))
      (progn
        (let ((comp-proc (get-buffer-process compilation-buffer-name)))
          (if comp-proc
	      (condition-case ()
		  (progn
		    (interrupt-process comp-proc)
		    (sit-for 1)
		    (delete-process comp-proc)
		    (when (and asy-compilation-auto-close
			       (eq asy-compilation-buffer 'none)
			       (not (eq asy-compilation-buffer 'visible)))
		      (sit-for 0.6)))
		(error ""))
	    ))
        (let ((view-inhibit-help-message t))
          (write-region "" 0 (asy-log-filename) nil))
        (compile command))
      (asy-compilation-wait pass auto-close))))

(defun asy-open-file(Filename)
  "Open the ps or pdf file Filename.
In unix-like system the variables `ps-view-command' and `pdf-view-command' are used.
In Windows the associated system file type is used instead."
  (let ((command
         (if running-unix-p
             (let ((ext (file-name-extension Filename)))
               (cond
                ((string= ext "ps") ps-view-command)
                ((string= ext "pdf") pdf-view-command)
                (t (error "Extension Not Supported."))))
           (asy-protect-file-name (file-name-nondirectory Filename))))
        )
    (if running-unix-p
        (start-process "" nil command Filename)
      (call-process-shell-command command nil 0))))

(defun lasy-TeX-master-file ()
  "Return the file name of the master file for the current document.
The returned string contain the directory but does not contain the extension of the file."
  (expand-file-name
   (concat (TeX-master-directory) (TeX-master-file nil t))))

(defun lasy-must-compile-p (TeX-Master-File out-file &optional Force)
  ""
  (or Force
      (file-newer-than-file-p
       (concat TeX-Master-File ".tex") out-file)
      (and (stringp (TeX-master-file)) ;; current buffer is not a mater tex file
           (file-newer-than-file-p buffer-file-name out-file))))

(defun lasy-view-ps (&optional Force  Filename fromtex)
  "Compile a LaTeX document embedding Asymptote code with latex->asy->latex->dvips and/or view the PostScript output.
If optional argument Force is t then force compilation."
  (interactive)
  (setq lasy-run-tex t)
  (setq lasy-compile-tex fromtex)
  (if (buffer-modified-p) (save-buffer))
  (when (eq asy-compilation-buffer 'never) (write-region "" 0 (asy-log-filename) nil))
  (let*
      ((b-b-n  (if Filename Filename (lasy-TeX-master-file)))
       (b-b-n-tex (asy-protect-file-name (concat b-b-n ".tex")))
       (b-b-n-ps (asy-protect-file-name (concat b-b-n ".ps")))
       (b-b-n-dvi (asy-protect-file-name (concat b-b-n ".dvi")))
       (b-b-n-asy (asy-protect-file-name (concat b-b-n ".asy")))
       (stderr (eq asy-compilation-buffer 'never)))
    (if (lasy-must-compile-p b-b-n (concat b-b-n ".ps") Force)
        (progn
          (let ((default-directory (file-name-directory b-b-n)))
            (asy-internal-compile (concat lasy-latex-command " " b-b-n-tex))
            (when (and (zerop asy-last-compilation-code) (file-readable-p (concat b-b-n ".asy")))
              (asy-internal-compile (concat asy-command-location lasy-command " " b-b-n-asy)  nil nil stderr)
              (when (zerop asy-last-compilation-code)
                (asy-internal-compile (concat lasy-latex-command " " b-b-n-tex))))
            (when (zerop asy-last-compilation-code)
              (asy-internal-compile (concat lasy-dvips-command " " b-b-n-dvi " -o " b-b-n-ps) nil t)
              (when (zerop asy-last-compilation-code)
                (asy-open-file (concat b-b-n ".ps"))))))
      (asy-open-file (concat b-b-n ".ps")))))

(defun lasy-view-pdf-via-pdflatex (&optional Force Filename fromtex)
  "Compile a LaTeX document embedding Asymptote code with pdflatex->asy->pdflatex and/or view the PDF output.
If optional argument Force is t then force compilation."
  (interactive)
  (setq lasy-run-tex t)
  (setq lasy-compile-tex fromtex)
  (if (buffer-modified-p) (save-buffer))
  (when (eq asy-compilation-buffer 'never) (write-region "" 0 (asy-log-filename) nil))
  (let*
      ((b-b-n  (if Filename Filename (lasy-TeX-master-file)))
       (b-b-n-tex (asy-protect-file-name (concat b-b-n ".tex")))
       (b-b-n-pdf (asy-protect-file-name (concat b-b-n ".pdf")))
       (b-b-n-asy (asy-protect-file-name (concat b-b-n ".asy")))
       ;; (stderr (or (eq asy-compilation-buffer 'never) lasy-compile-tex)))
       (stderr (eq asy-compilation-buffer 'never)))
    (if (lasy-must-compile-p b-b-n (concat b-b-n ".pdf") Force)
        (progn
          (let ((default-directory (file-name-directory b-b-n)))
            (asy-internal-compile (concat lasy-pdflatex-command " " b-b-n-tex))
            (when (and (zerop asy-last-compilation-code) (file-readable-p (concat b-b-n ".asy")))
              (asy-internal-compile (concat asy-command-location lasy-command " " b-b-n-asy) nil nil stderr)
              (when (zerop asy-last-compilation-code)
                (asy-internal-compile (concat lasy-pdflatex-command " " b-b-n-tex) t)))
            (when (zerop asy-last-compilation-code)
              (asy-open-file (concat b-b-n ".pdf")))))
      (asy-open-file (concat b-b-n ".pdf")))))

(defun lasy-view-pdf-via-ps2pdf (&optional Force Filename fromtex)
  "Compile a LaTeX document embedding Asymptote code with latex->asy->latex->dvips->ps2pdf14 and/or view the PDF output.
If optional argument Force is t then force compilation."
  (interactive)
  (setq lasy-run-tex t)
  (setq lasy-compile-tex fromtex)
  (if (buffer-modified-p) (save-buffer))
  (when (eq asy-compilation-buffer 'never) (write-region "" 0 (asy-log-filename) nil))
  (let*
      ((b-b-n  (if Filename Filename (lasy-TeX-master-file)))
       (b-b-n-tex (asy-protect-file-name (concat b-b-n ".tex")))
       (b-b-n-ps (asy-protect-file-name (concat b-b-n ".ps")))
       (b-b-n-dvi (asy-protect-file-name (concat b-b-n ".dvi")))
       (b-b-n-pdf (asy-protect-file-name (concat b-b-n ".pdf")))
       (b-b-n-asy (asy-protect-file-name (concat b-b-n ".asy")))
       ;; (stderr (or (eq asy-compilation-buffer 'never) lasy-compile-tex)))
       (stderr (eq asy-compilation-buffer 'never)))
    (if (lasy-must-compile-p b-b-n (concat b-b-n ".pdf") Force)
        (progn
          (let ((default-directory (file-name-directory b-b-n)))
            (asy-internal-compile (concat lasy-latex-command " " b-b-n-tex))
            (when (and (zerop asy-last-compilation-code) (file-readable-p (concat b-b-n ".asy")))
              (asy-internal-compile (concat asy-command-location lasy-command " " b-b-n-asy) nil nil stderr)
              (when (zerop asy-last-compilation-code)
                (asy-internal-compile (concat lasy-latex-command " " b-b-n-tex))))
            (when (zerop asy-last-compilation-code)
              (asy-internal-compile (concat lasy-dvips-pre-pdf-command " " b-b-n-dvi " -o " b-b-n-ps))
              (when (zerop asy-last-compilation-code)
                (asy-internal-compile (concat lasy-ps2pdf-command " " b-b-n-ps " " b-b-n-pdf) t)
                (when (zerop asy-last-compilation-code)
                  (asy-open-file (concat b-b-n ".pdf")))))))
      (asy-open-file (concat b-b-n ".pdf")))))

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
