from tkinter import *
from PIL import ImageTk, Image


def callback():
    print(e.get())
    L2.config(text="New text test test test test test test")


def makeentry(parent, caption, width=None, **options):
    Label(parent, text=caption).pack(side=LEFT)
    entry = Entry(parent, **options)
    if width:
        entry.config(width=width)
    entry.pack(side=LEFT)
    return entry


master = Tk()
master.configure(background="white")
master.geometry("1000x500+700+300")

b = Button(master, text="Enter", width=20, height=2, command=callback)
b.pack()
b.place(x=0, y=20)

e = Entry(master, width=50)
e.pack()
e.place(x=0, y=5)

e.focus_set()


canvas = Canvas(master, width=500, height=400)
canvas.pack()
img = ImageTk.PhotoImage(Image.open("girl.png"))
# img = PhotoImage(file="zoey.jpg")
canvas.create_image(20,0, anchor=NW, image=img)

L1 = Label(master, text="Zoey:", bg="white", font=("Helvetica", 13))
L1.pack()
L1.place(x=425, y=420)

L2 = Label(master, text="test", bg="white", font=("Helvetica", 13))
L2.pack()
L2.config(bg="white")
L2.place(x=475, y=420)

mainloop()
e = Entry(master, width=50)
e.pack()

text = e.get()



