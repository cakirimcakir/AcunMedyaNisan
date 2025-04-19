import pandas as pd #matemaatiksel işlemler için
from sklearn.preprocessing import LabelEncoder #Sayısal olmayan kategorileri sayısallaştırmak için
from sklearn.model_selection import train_test_split #Test-Eğitim verilerini ayırmak için
from sklearn.linear_model import LogisticRegression #Seçtiğimiz model Lojistik regresyon
from sklearn.metrics import accuracy_score, classification_report #sonuç gösteren metrikler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline


#Veri
df = pd.read_csv("sgdata.csv")
le = LabelEncoder()

df['Marital status'] = le.fit_transform(df['Marital status']) #non-single (divorced / separated / married / widowed-single=0-1
df['Education'] = le.fit_transform(df['Education']) #high school-other/unknown-university =0-1-2
df['Occupation'] = le.fit_transform(df['Occupation']) #management / self-employed / highly qualified employee / officer-skilled employee / official-unemployed / unskilled=0-1-2


# Veri yapısına genel bakış
print(df.shape)        # Kaç satır ve sütun var
print(df.columns)      # Sütun isimleri
print(df.dtypes)       # Türler
print(df.head())       # İlk 5 satır
print(df.isnull().sum()) #Toplam Eksik sütun sayısını gösterir

df["Income_Class"]=(df["Income"]>100000).astype(int) #Hedef değişkenimizi gelir olarak belirledik(yıllık) 100.000 üstü 1 kalanı 0'dır.

X=df.drop(["Income","Income_Class","ID"],axis=1)
y=df["Income_Class"]

X_train,X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42) #Eğitim ve test verisi olarak ayırıyoruz (80/20)

logistic_model=LogisticRegression(max_iter=1000) #modelimi lojistik regrresyon seçtik. 1000 iterasayon yapacak.
logistic_model.fit(X_train,y_train) #.fit eğitim için kullandık.

y_pred=logistic_model.predict(X_test) #Tahminleme

print("Doğruluk Oranı:" ,accuracy_score(y_test,y_pred))

print("\nSınıflandırma Raporu:\n", classification_report(y_test,y_pred))

#SVM Modeli
svm_model=make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
svm_model.fit(X_train,y_train) #Eğitiyoruz.

y_pred_svm=svm_model.predict(X_test) #tahminleme

print("SVM- Doğruluk Oranı:",accuracy_score(y_test,y_pred_svm))
print("\nSVM- Sınıflandırma Raporu:\n",classification_report(y_test,y_pred_svm))