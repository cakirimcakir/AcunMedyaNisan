from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd 
#Gerekli olan kütüphaneleri yüklüyoruz.
print("\nVeri Seti Yüklendi!")
diabetes = load_diabetes() # Veri setini yükleyelim
df_features=pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
df_features['target']=diabetes.target
#load.diabates(), kullandık. Bu fonksiyon veri setini bunch olarak döndürür.veri setindeki tüm verileri "data" içerisinde yer almış olur. Bu nedenle .data ve .feature_names kullanıyoruz.
x=df_features[["bmi"]]#Bağımsız değişkenimiz:bmi(vücut kitle endeksi)
y=df_features["target"]#Bağımlı değişkenimiz: target
print("\nVeri setimizi Eğitim ve Test Olarak Ayrılıyor..")
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)
#X,y ddğişkenlerimizin eğitim ve test verilerini tanımlıyoruz. oranımızı %80 eğitim, %20 test şeklinde oluşturuyoruz.

print(f"Eğitim Seti Boyutu:{x_train.shape}")
print(f"Test Seti Boyutu:{x_test.shape}")
#Eğitim ve test boyutunu .shape ile gösterir.

print("\n Lineer Regresyon Modeli Eğitiliyor..")

model=LinearRegression() #Lineer regresyon modeli oluşturup "model" e tanımlıyoruz.

model.fit(x_train,y_train) #.fit ile eğitim verilerini eğitiyoruz.


#R2 Skor Hesaplaması:

y_pred=model.predict(x_test) #bağımsız değişkenimizin test verilerini tahminleme yapıp y_pred değişkeninde saklıyoruz.
r2=r2_score(y_test,y_pred)
print(f"Basit Regresyon Modeli R2 Skoru:{r2:.4f}")#r2_score ile bağımlı değişkenimizin test ve tahmin verilerini eşleştirerek r2 skoruna ulaşıyoruz.

x_multiple=df_features.drop(columns=['target']) #targetı .drop ile çıkartarak bağımsız değişkenlerimizi tanımladık.

x_train_multiple,x_test_multiple,y_train,y_test=train_test_split(x_multiple,y,test_size=0.2, random_state=42) #eğitim ve test verilerini böldük.

model_multiple = LinearRegression() #çoklu model için lineer regresyon modeli oluşturduk
model_multiple.fit(x_train_multiple, y_train) #. fit kullanarak eğitim verilerimiz ile modeli eğitiyoruz.
y_pred_multiple = model_multiple.predict(x_test_multiple) #.predict kullanarak bağımsız değişkenimizin test verileriyle tahminleme yapıyoruz.
r2_multiple = r2_score(y_test, y_pred_multiple) #r2_score kullanarak bağımlı değişkenimizin test verileri, ve tahminleme verilerimizi kullanarak r2 skoruna ulaştık.
print(f"Çoklu Lineer Regresyon Modeli R² Skoru: {r2_multiple:.4f}") #Yazdırma İşlemi

#Hata Hesaplamaları
mae_multiple=mean_absolute_error(y_test,y_pred_multiple)
mse_multiple=mean_squared_error(y_test,y_pred_multiple)
print(f"Çoklu Regresyon Modeli MAE: {mae_multiple:.4f}")
print(f"Çoklu Regresyon Modeli MSE: {mse_multiple:.4f}")

#Sadece "bmi" yi bağımsız değişken olarak aldığımız takdirde modelin ancak %23 civarında doğru tahmin yapabildiğini gösteriyor. Bu da düşük bir oran. Tek metrik veriyi açıklamak için yeterli gelmemiştir.

#Çoklu modelde ise tüm bağımsız değişkenlerimizi ele aldık(bmi,age,sex vb...). Burada da %42 gibi yine çok sağlam olmayan bir sayıya ulaştık. Bu skor 1'e ne kadar yakın olursa modelin o kadar başarılı olduğunu gösterir.

#MAE: 42.79
#Bu sayı modelin tahminleri ile gerçek değeler arasındaki farkın ortalma %42.79 olduğunu gösteriyor.
#MSE:2900
#Modelin hatalarının karelerini hesaplar.büyük hatalar kare işlemi dolayısıyla daha büyük eyki gösterir. rakam ne kdar yüksekse hata oranı o kadar yüksek manasına gelmektedir.