from pyjokes import get_joke
from tkinter import ttk
import tkinter as tk


def on_click():
    joke = get_joke()
    label.config(text = joke)

root = tk.Tk()
root.title('The Trickster')
root.geometry('1000x500')
label = ttk.Label(root, text = 'Hey, do you want to hear a joke?', justify = 'center')
label.pack(pady = 10)
btn1 = ttk.Button(root, text = chr(0x1F504), command = on_click)
btn1.pack(pady = 10)
btn2 = ttk.Button(root, text = 'You Done?', command = root.destroy)
btn2.pack(pady = 10)

root.mainloop()
