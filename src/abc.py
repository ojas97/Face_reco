import glob
path = '//home//ojas//PycharmProjects//Term_Project//src//Images//*.jpg'
files = glob.glob(path)
known_face_encoding=[]
for imag in files:
    name=imag[57:-4]
    print name
