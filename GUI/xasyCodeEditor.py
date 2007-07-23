#!/usr/bin/env python
#
# Author: Orest Shardt
# Created: June 29, 2007
#
from Tkinter import *
import tkMessageBox
import tkFileDialog
class xasyCodeEditor(Toplevel):
  """A dialog for editing an asy script"""
  def __init__(self,parent,text):
    """Initialize the dialog with a parent, initial text, and an update callback"""
    Toplevel.__init__(self,parent)
    self.title("Xasy Code Editor")
    self.transient(parent)
    self.focus_set()
    self.wait_visibility()
    self.grab_set()
    self.text = text.rstrip()

    self.mainMenu = Menu(self)
    self.config(menu=self.mainMenu)
    self.fileMenu = Menu(self.mainMenu)
    self.fileMenu.add_command(label="Open File",command=self.loadFile)
    self.fileMenu.add_separator()
    self.fileMenu.add_command(label="Apply changes and close",command=self.applyChangesAndQuit)
    self.fileMenu.add_command(label="Close",command=self.destroy)
    self.mainMenu.add_cascade(label="File",menu=self.fileMenu)

    self.editMenu = Menu(self.mainMenu)
    self.editMenu.add_command(label="Undo",command=self.editUndo)
    self.editMenu.add_command(label="Redo",command=self.editRedo)
    self.editMenu.add_separator()
    self.editMenu.add_command(label="Copy",command=self.editCopy)
    self.editMenu.add_command(label="Cut",command=self.editCut)
    self.editMenu.add_command(label="Paste",command=self.editPaste)
    self.editMenu.add_separator()
    self.editMenu.add_command(label="Select All",command=self.editSelectAll)
    self.mainMenu.add_cascade(label="Edit",menu=self.editMenu)

    self.rowconfigure(0,weight=1)
    self.columnconfigure(0,weight=1)
    self.textVScroll = Scrollbar(self,orient=VERTICAL)
    self.textHScroll = Scrollbar(self,orient=HORIZONTAL)
    self.textVScroll.grid(row=0,column=1,sticky=N+S)
    self.textHScroll.grid(row=1,column=0,sticky=E+W)
    self.textBox = Text(self,wrap=NONE,yscrollcommand=self.textVScroll.set,xscrollcommand=self.textHScroll.set,maxundo=200,undo=True)
    self.textBox.insert(1.0,self.text)
    self.textBox.grid(row=0,column=0,sticky=N+S+E+W)
    self.textVScroll.config(command=self.textBox.yview)
    self.textHScroll.config(command=self.textBox.xview)

  def getText(self):
    #wait to be done
    self.wait_window(self)
    return self.text

  def destroy(self):
    """Do any predestruction checking"""
    result = "yes"
    if self.textBox.get(1.0,END).rstrip() != self.text:
      result = tkMessageBox._show("Script modified","Update changes before quitting?",icon=tkMessageBox.QUESTION,type=tkMessageBox.YESNOCANCEL)
      if str(result) == "yes":
        self.text = self.textBox.get(1.0,END).rstrip()
    if str(result) != "cancel":
      Toplevel.destroy(self)

  def loadFile(self):
    """Load a file into the textBox"""
    file=tkFileDialog.askopenfile(filetypes=[("asy files","*.asy"),("All files","*")],title="Open File")
    if file != None:
      try:
        lines = file.read()
        self.textBox.delete(1.0,END)
        self.textBox.insert(END,lines)
        file.close()
      except:
        tkMessageBox.showerror("File Opening Failed.","Error in input file.")

  def editUndo(self):
    """Undo an edit in the textBox"""
    self.textBox.edit_undo()

  def editRedo(self):
    """Redo an undone edit in the textBox"""
    self.textBox.edit_redo()

  def editCopy(self):
    """TODO: implement Cut/Paste"""
    pass

  def editCut(self):
    """TODO: implement Cut/Paste"""
    pass

  def editPaste(self):
    """TODO: implement Cut/Paste"""
    pass

  def editSelectAll(self):
    """TODO: allow a select all command"""
    pass

  def applyChangesAndQuit(self):
    """Call the callback and close the dialog"""
    self.text = self.textBox.get(1.0,END).rstrip()
    self.destroy()

if __name__ == '__main__':
  #run a test
  root=Tk()
  ce = xasyCodeEditor(root,"Here is some text to edit")
  root.mainloop()
