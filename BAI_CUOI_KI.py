import numpy as np
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold




df = pd.read_csv('./IRIS.csv') 
#mã hóa nhãn y thành số (0,1,2)
# Tạo ánh xạ
custom_mapping = {
    "Iris-setosa": 0,
    "Iris-virginica": 1,
    "Iris-versicolor": 2,
    # Thêm các ánh xạ cho các giá trị chữ khác nếu cần
}
df['species'] = df['species'].map(custom_mapping)


#id3
class ID3():
    def __init__(self):
        self.tree = None
    
    # tính entropy theo class
    def ent(self, df) -> float:
        return sum([-i*math.log(i) for i in df[df.columns[-1]].value_counts(normalize=True).values if i!=0])
    
    # tính entropy theo thuộc tính (attribute) + class
    def  ent_atb(self, df, atb: str) -> float:
        return abs(sum([self.ent(df[[atb, df.columns[-1]]][df[atb]==i]) * len(df[atb][df[atb]==i])/len(df) for i in df[atb].unique()]))

    # tính thông tin gain
    # đầu ra là tên thuộc tính có giá trị gain lớn nhất
    def gain(self, df) -> str:
        return df.columns[:-1][np.argmax([self.ent(df) - self.ent_atb(df, i) for i in df.columns[:-1]])]


    # lọc lấy bảng con chứa 1 loại thông tin trong 1 trường dữ liệu
    def sub_df(self, df, node, value):
        return df[df[node] == value].reset_index(drop=True)

    # xây dựng cây
    def buildTree(self, df, tree = None):
        node = self.gain(df)
        attValue = np.unique(df[node])
   
        if tree is None:                    
            tree = {}
            tree[node] = {}

        for value in attValue:
            subtable = self.sub_df(df, node, value)
            labels, counts = np.unique(subtable[df.columns[-1]],return_counts=True) 
            if len(counts) == 1:
                tree[node][value] = labels[0]                                                    
            else:        
                tree[node][value] = self.buildTree(subtable)
        return tree
        
    # fit mô hình
    def fit(self, x_train, y_train):
        df = pd.concat([x_train, y_train], axis=1)
        self.tree = self.buildTree(df)

    # dự đoán
    def predict(self, df):
        return np.array([int(self.pred(self.tree, i)) for _, i in df.iterrows()])

    def pred(self, tree, sample) -> str:
        if not isinstance(tree, dict):
            return tree
        else:
            root_node = next(iter(tree))
            feature_value = sample[root_node]
            if feature_value in tree[root_node]:
                return self.pred(tree[root_node][feature_value], sample)
            else:
                feature_value = list(tree[root_node].keys())[0]
                return self.pred(tree[root_node][feature_value], sample)



# tinh error, y thuc te, y_pred: dl du doan
def error(y,y_pred):
    sum=0
    for i in range(0,len(y)):
        sum = sum + abs(y[i] - y_pred[i])
    return sum/len(y)  # tra ve trung binh


##tinh accuracy
def accuracy_score(y_test, y_pred):
    return np.sum(np.equal(y_test, y_pred))/len(y_test)

#danh gia chat luong model
from sklearn.metrics import precision_score,recall_score,f1_score

# print('Precission:',precision_score(y_test, y_pred, average='micro'))
# print('Recall:',recall_score(y_test, y_pred, average='micro'))
# print('F1_score',f1_score(y_test, y_pred, average='micro'))
# accuracy=accuracy_score(y_test, y_pred)
# print('accuracy:',accuracy)

min_ID3=999999
min_CART=999999
k = 3
i=0
##########ID3
##su dung k fold de tim model tot nhat
y_test_ID3=df
y_pred_ID3=df
kf = KFold(n_splits=k, random_state=None)
for train_index, validation_index in kf.split(df):
    x_train, x_validation = df.iloc[train_index,:4], df.iloc[validation_index, :4]
    y_train, y_validation = df.iloc[train_index, 4], df.iloc[validation_index, 4]

    #khoi tao model ID3 huan luyenmo hinh
    model = ID3()
    model.fit(x_train,y_train)
    #tinh error
    y_train_pred = model.predict(x_train)
    y_validation_pred = model.predict(x_validation)
    y_train = np.array(y_train)
    y_validation = np.array(y_validation)

    sum_error = error(y_train,y_train_pred)+error(y_validation, y_validation_pred)
    #kiem tra model
    if(sum_error < min_ID3):
        min = sum_error
        tree_classified_ID3=model
        y_pred_ID3=y_validation_pred
        y_test_ID3=y_validation
        



##################CART
y_test_CART=df
y_pred_CART=df
for train_index, validation_index in kf.split(df):
    x_train, x_validation = df.iloc[train_index,:4], df.iloc[validation_index, :4]
    y_train, y_validation = df.iloc[train_index, 4], df.iloc[validation_index, 4]

   #khoi tao model Cart + huan luyen mo hinh
    model = DecisionTreeClassifier(criterion='gini',max_depth=2)
    model.fit(x_train,y_train)
    #tinh error
    y_train_pred = model.predict(x_train)
    y_validation_pred = model.predict(x_validation)
    y_train = np.array(y_train)
    y_validation = np.array(y_validation)
    sum_error_CART = error(y_train,y_train_pred)+error(y_validation, y_validation_pred)
    #kiem tra mo hinh
    if(sum_error_CART < min_CART):
        min_CART = sum_error_CART
        tree_classified_CART=model
        y_test_CART=y_validation
        y_pred_CART=y_validation_pred








################GIAO DIEN######################
from tkinter import messagebox


import tkinter as ttk

def id3():
    if(entry_sepal_length.get()=="" or entry_sepal_width.get()=="" or entry_petal_length.get()=="" or entry_petal_width.get()==""):
        messagebox.showinfo("Thông báo", "Dữ liệu phải được nhập đầy đủ!")
    else:
        label_result_id3.delete("1.0",ttk.END)
        text="Precission:" + str(precision_score(y_test_ID3, y_pred_ID3, average='micro'))+"\nRecall : " +str( recall_score(y_test_ID3, y_pred_ID3, average='micro'))+"\nF1 : " +str( f1_score(y_test_ID3, y_pred_ID3, average='micro'))+"\nAccuracy:" +str( accuracy_score(y_test_ID3, y_pred_ID3))
        label_result_id3.insert( ttk.END,"id3:\n"+text)
def cart():
    if(entry_sepal_length.get()=="" or entry_sepal_width.get()=="" or entry_petal_length.get()=="" or entry_petal_width.get()==""):
        messagebox.showinfo("Thông báo", "Dữ liệu phải được nhập đầy đủ!")
    else:
        label_result_cart.delete("1.0", ttk.END)
        text="\nPrecission:" + str(precision_score(y_test_CART, y_pred_CART, average='micro'))+"\nRecall : " +str( recall_score(y_test_CART, y_pred_CART, average='micro'))+"\nF1 : " +str( f1_score(y_test_CART, y_pred_CART, average='micro'))+"\nAccuracy:" +str( accuracy_score(y_test_CART, y_pred_CART))
        label_result_cart.insert( ttk.END,"cart"+text)




def submit():
    #du doan x thuoc nhom may
    #lay du lieu tu textbox de du doan
    sepal_length=entry_sepal_length.get()
    sepal_width=entry_sepal_width.get()
    petal_length=entry_petal_length.get()
    petal_width=entry_petal_width.get()
    sample_data=pd.Series({
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width,
    })
    # #du doan cua id3
    # print("thuoc nhom(su dung id3) :",tree_classified_ID3.predict(pd.DataFrame([sample_data]))[0])
    # #du doan cua cart
    # print("thuoc nhom(su dung cart) :",tree_classified_CART.predict(pd.DataFrame([sample_data]))[0])

    ##so sanh
    accuracy_CART= accuracy_score(y_test_CART, y_pred_CART)
    accuracy_ID3= accuracy_score(y_test_ID3,y_pred_ID3)
    model = tree_classified_CART
    accuracy_score_best=accuracy_CART
    name = "CART"
    if(accuracy_score_best <= accuracy_ID3):
        accuracy_score_best = accuracy_ID3
        model = tree_classified_ID3
        name = "ID3"
    new_predict=model.predict(pd.DataFrame([sample_data]))[0]
    print("thuoc nhom ( su dung"+ name +") :",new_predict)
    text=""
    if(new_predict==0):
        label_result_submit.delete("1.0",ttk.END)
        text=str("Đặc điểm loài hoa giống với họ ( su dung"+ name +") : Iris-setosa\n")
        label_result_submit.insert( ttk.END,text)
    elif(new_predict==1):
        label_result_submit.delete("1.0",ttk.END)
        text=str("Đặc điểm loài hoa giống với họ ( su dung"+ name +") : Iris-virginica\n")
        label_result_submit.insert( ttk.END,text)
    else:
        label_result_submit.delete("1.0",ttk.END)
        text=str("Đặc điểm loài hoa giống với họ ( su dung"+ name +") : Iris-versicolor\n")
        label_result_submit.insert( ttk.END,text)


def validate_number_input(action, value_if_allowed):
    if action == '1':  # Kiểm tra xem có phải là lần nhập dữ liệu mới hay không
        if value_if_allowed.replace('.', '', 1).isdigit():  # Kiểm tra xem giá trị nhập có phải là số
            return True
        else:
            return False
    return True
    

window = ttk.Tk()
window.title('Bài kết thúc môn nhóm 10: Dự đoán hoa ')
window.geometry('800x600')
window.configure(background="#dcdde1")
fram1 = ttk.Frame(window,background="#dcdde1")

label_login = ttk.Label(fram1,text="Thuật toán ID3,Cart",font=("Tahoma",20),fg="#192a56",background="#dcdde1")

label_sepal_length = ttk.Label(fram1,text="Độ dài đài hoa: ",fg="#182C61",bg="#dcdde1",font=("Tahoma",13))
# Tạo một ô nhập (Entry) và đặt kiểu dữ liệu là số
validate_number = window.register(validate_number_input)
entry_sepal_length = ttk.Entry(fram1, width=40,validate="key",validatecommand=(validate_number, '%d', '%P'))

label_sepal_width = ttk.Label(fram1,text="Độ rộng đài hoa: ",fg="#182C61",bg="#dcdde1",font=("Tahoma",13))
entry_sepal_width = ttk.Entry(fram1, width=40,validate="key",validatecommand=(validate_number, '%d', '%P'))

label_petal_length = ttk.Label(fram1,text="Độ dài cánh hoa: ",fg="#182C61",bg="#dcdde1",font=("Tahoma",13))
entry_petal_length = ttk.Entry(fram1, width=40,validate="key",validatecommand=(validate_number, '%d', '%P'))

label_petal_width = ttk.Label(fram1,text="Độ rộng cánh hoa: ",fg="#182C61",bg="#dcdde1",font=("Tahoma",13))
entry_petal_width = ttk.Entry(fram1, width=40,validate="key",validatecommand=(validate_number, '%d', '%P'))

btn_calculate_id3= ttk.Button(fram1,bg="#1B9CFC",text="Phương pháp ID3 ",width="5",height="1",font=("Tahoma",13),cursor="hand2",activebackground="#457593",command=id3)
btn_calculate_cart= ttk.Button(fram1,bg="green",text="Phương pháp Cart",width="5",height="1",font=("Tahoma",13),cursor="hand2",activebackground="#457593",command=cart)
label_result_id3 = ttk.Text(fram1,font=("Tahoma",10), width=40, height=10,fg="#192a56",background="#dcdde1",state="normal")
label_result_cart = ttk.Text(fram1, width=40, height=10,font=("Tahoma",10),fg="#192a56",background="#dcdde1",state="normal")

btn_calculate_submit= ttk.Button(fram1,bg="#1B9CFC",text="Dự đoán",width="5",height="1",font=("Tahoma",13),cursor="hand2",activebackground="#457593",command=submit)
label_result_submit = ttk.Text(fram1, width=40, height=10,font=("Tahoma",10),fg="#192a56",background="#dcdde1",state="normal")

# Áp dụng tag cho từ cần gạch chân (từ "gạch chân")

label_login.grid(row=0,column=1,columnspan=2,pady=20)

label_sepal_length.grid(row=1,column=0,pady=10,sticky="w")
entry_sepal_length.grid(row=1,column=1,pady=10)

label_sepal_width.grid(row=1,column=2,pady=10,padx=10,sticky="w")
entry_sepal_width.grid(row=1,column=3,pady=10)

label_petal_length.grid(row=2,column=0,pady=10,sticky="w")
entry_petal_length.grid(row=2,column=1,pady=10)

label_petal_width.grid(row=2,column=2,pady=10,padx=10,sticky="w")
entry_petal_width.grid(row=2,column=3,pady=10)

btn_calculate_id3.grid(row=3,column=0,pady=10,padx=2,sticky="new")

btn_calculate_cart.grid(row=3,column=2,pady=10,padx=10,sticky="new")


label_result_id3.grid(row=4,column=0,pady=10,padx=10,sticky="new",columnspan=2)

label_result_cart.grid(row=4,column=2,pady=10,padx=10,sticky="new",columnspan=3)

btn_calculate_submit.grid(row=5,column=0,pady=10,padx=10,sticky="new")

label_result_submit.grid(row=5,column=1,pady=10,padx=10,sticky="new",columnspan=2)

fram1.pack()


window.mainloop()



