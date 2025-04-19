import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import matplotlib as plt
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
#Gerkli olan kütüphaneler

df=pd.read_csv("sgdata.csv") #verisetini okutuyoruz.
le=LabelEncoder() #

df['Marital status']=le.fit_transform(df['Marital status']) #non-single (divorced / separated / married / widowed-single=0-1
df['Education']=le.fit_transform(df['Education']) #high school-other/unknown-university =0-1-2
df['Occupation']=le.fit_transform(df['Occupation']) #management / self-employed / highly qualified employee / officer-skilled employee / official-unemployed / unskilled=0-1-2

df["Income_Class"]= (df["Income"]>100000).astype(int) #ıncome class oluşturup 100.000 üzerini 1 kalanını 0 olarak int yapıyoruz.
X=df.drop(["Income","Income_Class","ID"],axis=1) #Belirli sütunları çıkartıp bağımsız değişkenlerimizi belirledik
y=df["Income_Class"] #Bağımlı değişkenimiz

X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42) #eğitim ve test verilerimiz

tree_model=DecisionTreeClassifier(random_state=42, class_weight="balanced") #Ağaç modeli kullanıp veri dengesini sağlamak amacıyla balanced kullandık
tree_model.fit(X_train,y_train) #modeli eğitiyoruz.

y_pred_tree=tree_model.predict(X_test) #tahminleme

print("Karar Ağacı-Doğruluk Oranı:",accuracy_score(y_test,y_pred_tree))
print("\nKarar Ağacı-Sınıflandırma Raporu:\n", classification_report(y_test,y_pred_tree))      

#Karışıklık Matrisi
model = DecisionTreeClassifier(max_depth=5, random_state=42)

cm = confusion_matrix(y_test, y_pred_tree) #test verisi, test üzerine yapılan tahminler
plt.figure(figsize=(6, 4)) #grafik boyutu
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Düşük Gelir", "Yüksek Gelir"], yticklabels=["Düşük Gelir", "Yüksek Gelir"]) #ısı haritası
plt.xlabel('Tahmin Edilen') #x etiketi 
plt.ylabel('Gerçek') #y etiketi
plt.title('Confusion Matrix - Karar Ağacı') #ana başlık
plt.show() #gösterme fonksiyonu

# SMOTE ile veri dengele
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Yeni model
tree_model_smote = DecisionTreeClassifier(random_state=42)
tree_model_smote.fit(X_res, y_res)

# Tahmin yap
y_pred_smote = tree_model_smote.predict(X_test)

# Performans
print("Karar Ağacı (SMOTE) - Doğruluk Oranı:", accuracy_score(y_test, y_pred_smote))
print("\nKarar Ağacı (SMOTE) - Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_smote))
print("\nKarar Ağacı (SMOTE) - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_smote))
print(df["Income_Class"].value_counts())