from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten
import tensorflow as tf
from tkinter import Label, Tk, Button, font
from tkinter import filedialog
from PIL import Image, ImageTk


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
    label_viuda['text'] = ''
    label_violin['text'] = ''
    label_tarantula['text'] = ''
    prob_viuda = round(float(result[0][0]),3) * 100
    prob_violinista = round(float(result[0][1]),3) * 100
    prob_tarantula = round(float(result[0][2]),3) * 100
    label_viuda['text'] = f'Probabilidad de ser de la familia de las viudas: {prob_viuda}%'
    label_violin['text'] = f'Probabilidad de ser araña violinista: {prob_violinista}%'
    label_tarantula['text'] = f'Probabilidad de ser una tarántula: {prob_tarantula}%'

    image_spider = Image.open(filename)
    image_spider = image_spider.resize((450,450))
    test = ImageTk.PhotoImage(image_spider) 
    label1 = Label(image=test)
    label1.image = test  
    label1.place(x=700, y=100)
if __name__ == '__main__':
    root = Tk()
    root.geometry('1280x720')
    root.title('Clasificador de arañas')

    Button(root,text='Seleccionar imagen',font=font.Font(size=15),command=load_image).place(x=600,y=30)
    label_viuda = Label(root,text=f'',font=font.Font(size=15))
    label_viuda.place(x=0,y=100)
    label_violin = Label(root,text=f'',font=font.Font(size=15))
    label_violin.place(x=0,y=200)
    label_tarantula = Label(root,text=f'',font=font.Font(size=15))
    label_tarantula.place(x=0,y=300)
    btn_predict = Button(root,text='Realizar predicción',font=font.Font(size=15),command=predict_result,state='disabled')
    btn_predict.place(x=600,y=600)
    root.mainloop()