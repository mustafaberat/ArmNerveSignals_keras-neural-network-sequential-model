#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Veri Açıklaması: Bir insanın kolundaki sinir sinyallerine göre elini açması
# veya elini kapatması

## Importing the libraries

## numpy modülü array çarpımları gibi yüksek işlem gücü isteyen lineer cebir
# işlemlerini bize fonksiyonel olarak sunan bir modül

## pandas modülü verimiz üzerinde işlem yapmamızı sağlayan bir modül. Veriyi
# çekme, ayırma vb. gibi işlemler için kullanılıyor.
import numpy as np
import pandas as pd

## Importing the dataset

## Verisetimizi projemize yüklüyoruz. .csv olması read_csv() fonksiyonunu
# kullandığımız için önemli.
dataset = pd.read_csv('Emg_kucuk_dataset.csv')

## Burada veriyi anaconda'da açıp inceledikten sonra gerekli feature'lara karar
# verip, gereksizleri ele almama ve class değişkenine karar verme işi yapılıyor.
# Bu kararlardan sonra 1-5'in feature, 5'in de class olacağı
# görülüyor. Alttaki satırda da veri setinin 1-5 arası alınıyor.
X = dataset.iloc[:, 1:5].values

## 5. kolon da class olarak alınıyor.
y = dataset.iloc[:, 5].values

## Encoding categorical data

## Aşağıdaki kütüphaneden LabelEncoder sınıfını çağırarak "Encoding Categorical
# Data to Numerical Data" işlemini gerçekleştireceğiz.
from sklearn.preprocessing import LabelEncoder

## LabelEncoder sınfından oluşturduğumuz nesne ile y veri seti yani dizisini
# kolayca fit_transform() metoduna sokarak numerical veriye çeviriyoruz. Burada
# metoda parametre olarak y'yi veriyoruz. Geri dönüşü de yine aynı şekilde
# kendisine eşitliyoruz.
labelencoder_y_1 = LabelEncoder()
y = labelencoder_y_1.fit_transform(y)

## Feature Scalling

## Veri önişlemenin son aşamalarından olan feature sclaing yapıyoruz. Veri
# setimizde diğer kolonlara baskın çıkabilecek sayısal değerlere sahip
# kolonlar var. Bu durumu bertaraf etmek için feature scale yöntemi olan
# standartlaştırma yöntemini kullanmak adına StandardScaler sınıfını import
# ediyor ve X veri setine uyguluyoruz. y veri setine feature scale uygulayıp
# uygulamamak ise fark edici bir nokta değil ama genelde uygulamak tercih
# edilen seçim
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

## Splitting the dataset into the Training set and Test set

## Tüm veri ön işleme aşamalarından sonra test-train ayrımı yapılıyor. Bunun için
# aşağıdaki kütüphanenin fonksiyonu kullanılıyor.
from sklearn.model_selection import train_test_split

## Fonksiyon 4 tane değer döndürüyor. Bunlar X yani feature kolonları için
# eğitim ve test, ile y yani class kolonu için eğitim ve test verileri.
# random_state parametresinin 1 olması bu parametreyi kullanıp 1 yapan herkese
# karışık ama aynı veri setinin geleceğini ifade ediyor.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)


## Artık ANN zamanı.Yapay Sinir Ağı kodlamak için keras kütüphanesini import
# ediyoruz. Bilgisayarımıza terminale 'pip install keras' yazarak kütüphaneyi
# yükleyebiliriz. YSA kodlamak için models ve layers modüllerini de import
# ediyoruz.
import keras
from keras.models import Sequential
from keras.layers import Dense

## models kütüphanesinden import ettiğimiz Sequential sınıfından nesne
# oluşturuyoruz. Bu kod ile genel YSA tanımlamasını yapıyoruz. Katman eklemekten,
# ağı eğitmeye, tahminleri ortaya çıkarmaya kadar tüm işlemleri bu classifier
# nesnesi ile yapacağız.
classifier = Sequential()

## classifier nesnesinden add metodu ile ilk katmanı ekliyoruz. add metodu içine
# Dense sınıfı ve bu sınıfın constructor yapısına gerekli parametreleri girerek
# ilk katmanımızı ekliyoruz. Keras kütüphanesinin
# mimarisinden dolayı YSA yapımıza giriş katmanı eklemek gibi bir durum yok.
# Direkt olarak ilk gizli katmanımızı eklemiş olduk. Ancak tabii ki ağa giriş input
# sayısını vermek gerekiyor, bunu da input_dim parametresi ile yaptık. Bu parametreyi
# sadece ilk gizli katmanı eklerken yazıyoruz. Gizli katman için de units parametresi
# ile gizli katman nöron sayısını, kernel_initializer parametresi ile ağırlık
# başlangıç değer atamasını, activation parametresi ile aktivasyon fonksiyonu
# seçimini yaptık.
classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))

# İkinci gizli katmanımızı da aynı şekilde ekliyoruz. Bu sefer söylediğimiz
# üzere input_dim parametresi yok.
classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))

# Son olarak çıkış katmanımızı ekliyoruz. Kodlama olarak bunu da aynı şekilde
# ekliyoruz, sadece units parametresini 1 yapıyoruz. Tahmin edeceğimiz değerler
# yani labelımız 0 ve 1'den oluşuyordu. İkili değer olduğu için tek çıkış nöronu
# yeterli oluyor.
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

## YSA için eğitimde gerekli diğer hiperparametreleri belirleme zamanı. optimizer
# parametresi öğrenme fonksiyonu seçimi için, loss parametresi loss fonksiyonu
# seçimi için kullanılıyor. metrics parametresi ise hata kriterini accuracy'e göre
# belirleyeceğimiz anlamına geliyor. Tüm bunları compile metodu ile yapıyoruz.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

## Artık eğitim zamanı. fit metodu ile eğitimi gerçekleştirceğiz. X_train ve
# y_train'i veriyoruz. batch_size, epochs parametrelerine de standart
# olarak tercih edilen değerleri giriyoruz
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


## Predicting the Test set results

## Algoritmanın eğitimi tamamlandı. Performansını ölçmek adına test için
# ayırdığımız eğitime karışmamış verileri modele veriyoruz ve bize test setindeki
# verilerin tahminlerini yapıyor.
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

## Making the Confusion Matrix

## Tahminleri yaptırdıktan sonra doğruluk oranımızı görmek ve modelimizin somut
# çıktısını almak adına Confusion Matrix'i hesaplatıyoruz. Hesaplatmak için
# çağırdığımız kütüphanedeki fonksiyona görüldüğü üzere test setinin gerçek
# verilerini ve modelin tahmin ettiği verileri veriyourz. Bu adım uygulanmadan
# önce öğrencilere Confusion Matrix anlatılabilir.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

## test verisetinde test ettiğimiz veriler dışında dışarıdan tek bir tane veri
# verip modelin tahminini görmek için yine predict metodunu kullanıyoruz ama
# bu sefer parametre olarak tahmin yaptırmak istediğimiz veriyi array şeklinde
# veriyoruz. Ayrıca veri ön işlemede tüm veriyi algoritmaya vermden önce son
# aşamada feature scale yaptığımız için burada da predict etmeden önce scale
# işlemine tabii tutuyoruz.
new_prediction = classifier.predict(sc.transform(np.array([[-11, -20, -3, 4]])))
new_prediction = (new_prediction > 0.5)
