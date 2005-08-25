;; Emacs mode file for editing Asymptote source files
;;
;; Uncomment and copy these two lines to your .emacs initialization file:
;;(autoload 'asy-mode "/usr/local/share/asymptote/asy-mode.el" "Asymptote major mode." t)
;;(setq auto-mode-alist (cons (cons "\\.asy$" 'asy-mode) auto-mode-alist))

(load "cc-mode")
(load "font-lock")

(defun asy-mode ()
  "Asymptote mode."
  (interactive)
  (c++-mode)
  (setq mode-name "Asymptote")
  (make-local-variable 'c++-font-lock-extra-types)
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
)

(global-font-lock-mode t)
(column-number-mode t)


