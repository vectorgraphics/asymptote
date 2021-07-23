(when (require 'cc-mode nil t)
  (require 'asy-mode nil t)

  (defvar my-c-style '((c-basic-offset . 2)
		       (c-tab-always-indent . nil)
                       (c-offsets-alist . ((innamespace nil)
                                           (inline-open nil)
                                           (case-label +)
                                           ))
                       (c-cleanup-list . (brace-else-brace
                                          brace-else-if-brace
                                          brace-catch-brace
                                          empty-defun-braces
                                          defun-close-semi))
                       (c-hanging-braces-alist . ((brace-list-open)
                                                  (brace-entry-open)
                                                  (statement-cont)
                                                  (substatement-open after)
                                                  (block-close . c-snug-do-while)
                                                  (extern-lang-open after)
                                                  (inline-open)
                                                  (inline-close)
                                                  (namespace-open after)))
                       (c-hanging-semi&comma-criteria . (c-semi&comma-no-newlines-for-oneline-inliners
                                                         c-semi&comma-no-newlines-before-nonblanks
                                                         c-semi&comma-inside-parenlist))
                       ))

  (setq c-mode-hook 'c++-mode)

  (defun c-mode-common-addn() "Additions to c-and-c++-mode."
    (c-add-style "jcb" my-c-style t)
;;    (c-toggle-auto-hungry-state 1)
    (auto-fill-mode)
    )

  (setq c-mode-common-hook 'c-mode-common-addn)
  )
