from PySide6 import QtWidgets as Qw


def error_msgbox(parent, msg: str, title: str = "Error"):
    msgbox = Qw.QMessageBox(Qw.QMessageBox.Icon.Critical, title, msg, parent=parent)
    msgbox.show()
