(load-file "cc-mode2.el")
(c++-mode)
(indent-region (search-forward "" nil nil 2) (point-max) nil)
(untabify (point-min) (point-max))
(delete-trailing-whitespace (point-min) (point-max))
(save-buffer)
