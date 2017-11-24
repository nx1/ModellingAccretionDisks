#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:10:16 2017

@author: yv
"""
import Tkinter as tk
from PIL import ImageTk, Image

#This creates the main frame of an application
frame = tk.Tk()
frame.title("Join")
frame.geometry("1200x800")
frame.configure(background='grey')

path = "LC/000000.png"

#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
img = ImageTk.PhotoImage(Image.open(path).resize((606,370),Image.ANTIALIAS))

#The Label widget is a standard Tkinter widget used to display a text or image on the screen.
panel = tk.Label(frame, image = img)

#The Pack geometry manager packs widgets in rows or columns.
panel.pack(side = "bottom", fill = "both", expand = "yes")

#Start the GUI
frame.mainloop()