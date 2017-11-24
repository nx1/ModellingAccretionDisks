#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:10:16 2017

@author: yv
"""
import Tkinter as tkinter

root = tkinter.Tk()

canvas = tkinter.Canvas(root)
canvas.pack(fill=tkinter.BOTH, expand=1) # expand

photo = tkinter.PhotoImage(file = '000000.png')
photo = photo.subsample(3)
root.geometry("450x450")
root.update()


img = canvas.create_image(10,10,anchor="nw", image=photo)

#root.after(20000, lambda: canvas.delete(img))

root.mainloop()