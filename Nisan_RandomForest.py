import pandas as pd #matematiksel ifadeler
from sklearn.preprocessing import LabelEncoder #sayısal olmayan kategorileri sayısallaştırır.
from sklearn.model_selection import train_test_split #eğitim ve test ayrımı için
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #skorlar için
import matplotlib.pyplot as plt #grafik için(karmaşıklık matrisi)
import seaborn as sns #ısı haritası için (karmaşıklık matrisi)

df=pd.read_csv("sgdata.csv")


le=LabelEncoder()
df['Marital status']=le.fit_transform(df["Marital status"])
df['Education']=le.fit_transform(df['Education'])
df["Occupation"]=le.fit_transform(df['Occupation'])

df["Income_Class"]= (df["Income"]>100000).astype(int) #hedef değişken. 100.000 üzeri 1 altı 0.

X=df.drop(["Income","Income_Class","ID"],axis=1) #BAĞIMSIZ DEĞİŞKENLER
y=df["Income_Class"] #BAĞIMLI DEĞİŞKENLER

X_train,X_test, y_train, y_test=train_test_split(X,y, test_size=0.2,random_state=42) #eğitim ve test boyutu

#Orman Modeli
rf_model=RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train,y_train) #Eğitim

y_pred_rf=rf_model.predict(X_test) #tahminleme

print("Random Forest-Doğruluk Oranı:",accuracy_score(y_test,y_pred_rf))
print("\nRandom Forest-Sınıflandırma Raporu:\n",classification_report(y_test,y_pred_rf))


#Karmaiıklık Matrisi
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=["Düşük Gelir", "Yüksek Gelir"], yticklabels=["Düşük Gelir", "Yüksek Gelir"])
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Confusion Matrix - Random Forest')
plt.show()

