from pyjokes import get_joke, get_jokes
from tkinter import ttk, Tk
import tkinter as tk

joke = get_joke()

root = Tk()
root.title('The Trickster')
frm = ttk.Frame(root, padding = 150)
frm.grid()
ttk.Label(frm, text = joke).grid(column = 0, row = 0)
ttk.Button(frm, text = 'You Done?', command = root.destroy).grid(column = 0, row = 1)

root.mainloop()
