#!/usr/bin/env python
#
# Author: Orest Shardt
# Created: June 29, 2007
#
from Tkinter import *
import tkMessageBox
import tkFileDialog
class xasyCodeEditor:
  """A dialog for editing an asy script"""
  def __init__(self,master,text,callback):
    """Initialize the dialog with a parent, initial text, and an update callback"""
    master.title("Xasy Code Editor")
    self.parent = master
    self.callback = callback
    master.bind("<Destroy>",self.closeUpEvt)
    self.mainMenu = Menu(master)
    master.config(menu=self.mainMenu)
    self.fileMenu = Menu(self.mainMenu)
    self.fileMenu.add_command(label="Open File",command=self.loadFile)
    self.fileMenu.add_separator()
    #self.fileMenu.add_command(label="Update",command=self.invokeCallback)
    #self.fileMenu.add_separator()
    self.fileMenu.add_command(label="Apply changes and quit",command=self.applyChangesAndQuit)
    self.fileMenu.add_command(label="Exit",command=self.closeUp)
    self.mainMenu.add_cascade(label="File",menu=self.fileMenu)

    self.editMenu = Menu(self.mainMenu)
    self.editMenu.add_command(label="Undo",command=self.editUndo)
    self.editMenu.add_command(label="Redo",command=self.editRedo)
    self.editMenu.add_separator()
    self.editMenu.add_command(label="Cut",command=self.editCut)
    self.editMenu.add_command(label="Paste",command=self.editPaste)
    self.editMenu.add_separator()
    self.editMenu.add_command(label="Select All",command=self.editSelectAll)
    self.mainMenu.add_cascade(label="Edit",menu=self.editMenu)

    master.rowconfigure(0,weight=1)
    master.columnconfigure(0,weight=1)
    self.textVScroll = Scrollbar(master,orient=VERTICAL)
    self.textHScroll = Scrollbar(master,orient=HORIZONTAL)
    self.textVScroll.grid(row=0,column=1,sticky=N+S)
    self.textHScroll.grid(row=1,column=0,sticky=E+W)
    self.textBox = Text(master,wrap=NONE,yscrollcommand=self.textVScroll.set,xscrollcommand=self.textHScroll.set,maxundo=200,undo=True)
    self.textBox.insert(END,text)
    self.textBox.grid(row=0,column=0,sticky=N+S+E+W)
    self.textVScroll.config(command=self.textBox.yview)
    self.textHScroll.config(command=self.textBox.xview)

    #remove this if the edit_modified function is fixed
    self.modified = True

  def loadFile(self):
    """Load a file into the textBox"""
    file=tkFileDialog.askopenfile(filetypes=[("asy files","*.asy"),("All files","*")],title="Open File",parent=self.parent)
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

  def editCut(self):
    """TODO: implement Cut/Paste"""
    pass

  def editPaste(self):
    """TODO: implement Cut/Paste"""
    pass

  def editSelectAll(self):
    """TODO: allow a select all command"""
    pass

  def closeUpEvt(self,event):
    """Close up the dialog"""
    self.parent.destroy()

  def invokeCallback(self):
    """Call the callback, providing the entered text"""
    self.callback(self.textBox.get(1.0,END))

  def applyChangesAndQuit(self):
    """Call the callback and close the dialog"""
    self.invokeCallback()
    self.modified = False
    self.closeUp()

  def closeUp(self):
    """Close the dialog, checking for changes"""
    #this function is broken: self.textBox.edit_modified()
    if not self.modified:
      self.parent.destroy()
    else:
      #prompt user about desire to quit
      #and find out what to do with the changes
      result = tkMessageBox._show("Script modified","Update changes before quitting?",icon=tkMessageBox.QUESTION,type=tkMessageBox.YESNOCANCEL)
      if result == "yes":
        self.invokeCallback()
        self.parent.destroy()
      elif result == "no":
        self.parent.destroy()

def defCallback(text):
  print text
if __name__ == '__main__':
  #run a test
  root=Tk()
  ce = xasyCodeEditor(root,"Here is some text to edit",defCallback)
  root.mainloop()
