from tkinter import *
from tkinter import messagebox

def Chake():
   
    import tensorflow as tf 
    import pandas as pd 
    import numpy as np 
    import matplotlib.pyplot as plt 
    import keras 
    from keras.models import Sequential 
    from keras.layers import Dense 
    from sklearn.metrics import confusion_matrix 

    data = pd.read_csv("Heart Disease Data Set.csv") 
    data.head() 

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
    imputer = imputer.fit(data.iloc[:,2:3])
    data.iloc[:,2:3] = imputer.transform(data.iloc[:,2:3])

    X = data.iloc[:,:13].values 
    y = data["target"].values 

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train, y_test = train_test_split(X,y,test_size = 0.3 , random_state = 0 ) 

    from sklearn.preprocessing import StandardScaler 
    sc = StandardScaler() 
    X_train = sc.fit_transform(X_train) 
    X_test = sc.transform(X_test)

    classifier = Sequential() 
    classifier.add(Dense(activation = "relu", input_dim = 13,  
                         units = 8, kernel_initializer = "uniform")) 
    classifier.add(Dense(activation = "relu", units = 14,  
                         kernel_initializer = "uniform")) 
    classifier.add(Dense(activation = "sigmoid", units = 1,  
                         kernel_initializer = "uniform")) 
    classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy',  
                       metrics = ['accuracy'] ) 

    classifier.fit(X_train , y_train , batch_size = 8 ,epochs = 100  ) 

    y_pred1 = classifier.predict(sc.transform(np.array([[entry_age.get(),entry_sex.get(),entry_cp.get(),entry_rbp.get(),
                                                        entry_sc.get(),entry_fbs.get(),entry_re.get(),entry_mhr.get(),
                                                         entry_eia.get(),entry_op.get(),entry_sp.get(),entry_ca.get(),
                                                         entry_th.get()]])))
   # y_pred1 = (y_pred1 > 0.5)
    a = y_pred1 > 0.5
    output.insert(0,a)

    
def reset():
    entry_age.delete(0, END)
    entry_sex.delete(0, END)
    entry_cp.delete(0, END)
    entry_rbp.delete(0, END)
    entry_sc.delete(0, END)
    entry_fbs.delete(0, END)
    entry_re.delete(0, END)
    entry_mhr.delete(0, END)
    entry_eia.delete(0, END)
    entry_op.delete(0, END)
    entry_sp.delete(0, END)
    entry_ca.delete(0, END)
    entry_th.delete(0, END)
    output.delete(0,END)

    
def close():
    answer = messagebox.askquestion("Confirm Exit", "Are you sure?")
    if answer == "yes":
        root.destroy()


root = Tk()
root.title("Heart Disease Diagnosis")
root.geometry("1920x1080")
root.configure(bg ="SeaGreen1")

label_0 = Label(root, text="HEART DISEASE DIAGNOSIS", font=("Helvetica", 50,"bold"),bg ="SeaGreen1")
label_0.place(x=200, y=50)

label_age = Label(root, text="Age",font=("Cambria (Headings)", 10,"bold"),bg ="SeaGreen1")
label_age.place(x=200, y=230)
entry_age = Entry(root, bg = "peach puff")
entry_age.place(x=400, y=230)

label_sex = Label(root, text="Sex", font=("Cambria (Headings)", 10,"bold"),bg ="SeaGreen1")
label_sex.place(x=200, y=280)
entry_sex = Entry(root, bg = "peach puff")
entry_sex.place(x=400, y=280)

label_cp = Label(root, text="Cp", font=("Cambria (Headings)", 10,"bold"),bg ="SeaGreen1")
label_cp.place(x=200, y=330)
entry_cp = Entry(root, bg = "peach puff")
entry_cp.place(x=400, y=330)

label_rbp = Label(root, text="Resting Blood Presure", font=("Cambria (Headings)", 10,"bold"),bg ="SeaGreen1")
label_rbp.place(x=200, y=380)
entry_rbp = Entry(root, bg = "peach puff")
entry_rbp.place(x=400, y=380)

label_sc = Label(root, text="Serum Cholestoral", font=("Cambria (Headings)", 10,"bold"),bg ="SeaGreen1")
label_sc.place(x = 200,y = 430)
entry_sc = Entry(root, bg = "peach puff")
entry_sc.place(x=400,y=430)

label_fbs = Label(root, text="Fasting blood suger",font=("Cambria (Headings)", 10,"bold"),bg ="SeaGreen1")
label_fbs.place(x=200, y=480)
entry_fbs = Entry(root, bg = "peach puff")
entry_fbs.place(x=400, y=480)

label_re = Label(root, text="Resting electrocardiographic", font=("Cambria (Headings)", 10,"bold"),bg ="SeaGreen1")
label_re.place(x=200, y= 520)
entry_re = Entry(root, bg = "peach puff")
entry_re.place(x=400, y=520)

label_mhr = Label(root, text="Maximum heart rate", font=("Cambria (Headings)", 10,"bold"),bg ="SeaGreen1")
label_mhr.place(x=700, y= 230)
entry_mhr = Entry(root, bg = "peach puff")
entry_mhr.place(x=900, y=230)

label_eia = Label(root, text= "exercise induced angina(0/1)", font=("Cambria (Headings)", 10,"bold"),bg ="SeaGreen1")
label_eia.place(x=700, y= 280)
entry_eia = Entry(root, bg = "peach puff")
entry_eia.place(x=900, y=280)

label_op = Label(root, text= "Old peak", font=("Cambria (Headings)", 10,"bold"),bg ="SeaGreen1")
label_op.place(x=700, y= 330)
entry_op = Entry(root, bg = "peach puff")
entry_op.place(x=900, y=330)

label_sp = Label(root, text= "Slop", font=("Cambria (Headings)", 10,"bold"),bg ="SeaGreen1")
label_sp.place(x=700, y= 380)
entry_sp = Entry(root, bg = "peach puff")
entry_sp.place(x=900, y=380)

label_ca = Label(root, text= "Ca",font=("Cambria (Headings)", 10,"bold"),bg ="SeaGreen1")
label_ca.place(x=700, y= 430)
entry_ca = Entry(root, bg = "peach puff")
entry_ca.place(x=900, y=430)

label_th = Label(root, text= "Thal",font=("Cambria (Headings)", 10,"bold"),bg ="SeaGreen1")
label_th.place(x=700, y= 480)
entry_th = Entry(root, bg = "peach puff")
entry_th.place(x=900, y=480)

output = Entry(root, bg = "peach puff")
output.place(x= 900, y= 530)


chake = Button(root, text="Chake heart disease", font=("Cambria (Headings)", 10,"bold"), command=Chake, bg = "cyan")
chake.place(x=700, y=530)

reset = Button(root,text = "      Reset      ",font=("Cambria (Headings)", 10,"bold"), command = reset,bg = "cyan")
reset.place(x =500, y=580)

close = Button(root,text = "      Exit       ",font=("Cambria (Headings)", 10,"bold"), command = close,bg = "cyan")
close.place(x =650, y=580)

root.mainloop()


