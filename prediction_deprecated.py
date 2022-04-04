from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten
import tensorflow as tf
from tkinter import Label, Tk, Button, font
from tkinter import filedialog


filename = ''

def load_image():
    global filename
    filename = filedialog.askopenfilename(initialdir='./',title='Seleccionar un archivo',filetypes=(('JPG','*.jpg'),('all files',"*.*")))
    btn_predict['state'] = 'active'

def predict_result():

    model = tf.keras.models.Sequential()
    # Primer convolucion
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
    model.add(MaxPooling2D((2,2)))

    # Segunda convolucion
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D((2,2)))

    # Tercera convolucion
    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(MaxPooling2D((2,2)))
    # Cuarta convolucion
    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512,activation='relu'))
    model.add(Dense(3,activation='softmax'))
    model.load_weights('./spider_classification.hdf5')
    img = load_img(filename,target_size=(150,150))
    img = img_to_array(img)
    img = img.reshape(1,150,150,3)
    img = img.astype('float32')
    img = img/255.0
    result = model.predict(img)

    Label(root,text=f'Probabilidad de ser viuda: {round(result[0][0],3)}',font=font.Font(size=14)).place(x=0,y=100)
    Label(root,text=f'Probabilidad de ser violinista: {round(result[0][1],3)}',font=font.Font(size=14)).place(x=0,y=200)
    Label(root,text=f'Probabilidad de ser tarántula: {round(result[0][2],3)}',font=font.Font(size=14)).place(x=0,y=300)

if __name__ == '__main__':
    root = Tk()
    root.geometry('1280x720')
    root.title('Clasificador de arañas')

    Button(root,text='Seleccionar imagen',font=font.Font(size=14),command=load_image).place(x=600,y=30)
    btn_predict = Button(root,text='Realizar predicción',font=font.Font(size=14),command=predict_result,state='disabled')
    btn_predict.place(x=600,y=600)
    root.mainloop()