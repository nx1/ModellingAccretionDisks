# -*- coding: utf-8 -*-

from Tkinter import *
from PIL import Image,ImageTk
import os.path

root = Tk()
root.title('grapher')
root.geometry("1466x615")
frame = Frame(root)
frame.grid()



N_l = [1,5,10,30]
const_l = [1.1,1.5]
Q_factor_l = [0.005,0.025,0.1]
tMax_factor_l = [1.1,1.4]
H_R_l= [0.1,1.0]
M_0_start_l = [1.0,10.0]


a = IntVar()
b = IntVar()
c = IntVar()
d = IntVar()
e = IntVar()
f = IntVar()


def ShowChoice():
    filename = '%d%d%d%d%d%d.png' % (a.get(), b.get(), c.get(), d.get(), e.get(), f.get())
    print filename
    
    path = "variable_graphs/LC/%s" % (filename)
    path2 = "variable_graphs/PSD/%s" % (filename)
    

    if os.path.exists(path):
        #Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
        img = ImageTk.PhotoImage(Image.open(path).resize((606,370),Image.ANTIALIAS))
        img2 = ImageTk.PhotoImage(Image.open(path2).resize((606,370),Image.ANTIALIAS))
        
        #The Label widget is a standard Tkinter widget used to display a text or image on the screen.
        
        Label(frame, image = img).grid(row=6, column=10)
        Label(frame, text=path).grid(row=7, column=10)
        
        Label(frame, image = img2).grid(row=6, column=11)
        Label(frame, text=path2).grid(row=7, column=11)
    else:
        
        Label(frame, text=path+' does not exist', fg="red").grid(row=6, column=10)
        Label(frame, text=path2+' does not exist', fg="red").grid(row=6, column=11)
        
    root.mainloop()
        
    
    


Label(frame, text="N").grid(row=0,column=0)
Label(frame, text="const").grid(row=1,column=0)
Label(frame, text="Q").grid(row=2,column=0)
Label(frame, text="tMax").grid(row=3,column=0)
Label(frame, text="H/R").grid(row=4,column=0)
Label(frame, text="M0start").grid(row=5,column=0)



for i, value in enumerate(N_l, 0):
    Radiobutton(frame, indicatoron = 0, command=ShowChoice, text=value, variable=a, value=i).grid(row=0, column=i+1)
    
for i, value in enumerate(const_l, 0):
    Radiobutton(frame, indicatoron = 0, command=ShowChoice, text=value, variable=b, value=i).grid(row=1, column=i+1)

for i, value in enumerate(Q_factor_l, 0):
    Radiobutton(frame, indicatoron = 0, command=ShowChoice, text=value, variable=c, value=i).grid(row=2, column=i+1)

for i, value in enumerate(tMax_factor_l, 0):
    Radiobutton(frame, indicatoron = 0, command=ShowChoice, text=value, variable=d, value=i).grid(row=3, column=i+1)

for i, value in enumerate(H_R_l, 0):
    Radiobutton(frame, indicatoron = 0, command=ShowChoice, text=value, variable=e, value=i).grid(row=4, column=i+1)

for i, value in enumerate(M_0_start_l, 0):
    Radiobutton(frame, indicatoron = 0, command=ShowChoice, text=value, variable=f, value=i).grid(row=5, column=i+1)


    





root.mainloop()