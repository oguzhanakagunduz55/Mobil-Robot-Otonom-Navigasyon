# Sensör Füzyonu ve Lokalizasyon Kullanarak LiDAR Tabanlı Otonom Navigasyon

Bu proje, Bursa Teknik Üniversitesi Mekatronik Mühendisliği bölümü "Mobil Robotlar" dersi kapsamında Oğuzhan Akagündüz tarafından geliştirilmiştir.

## Proje Hakkında
Bu çalışmada, 2B düzlemde modellenmiş karmaşık bir ortamda hareket eden mobil bir robot için otonom navigasyon ve lokalizasyon algoritmaları geliştirilmiştir. Robot, "Non-holonomic" kinematik modele sahip olup, engellerden kaçınmak için teğetsel kaydırma kuvveti (swirl force) destekli Yapay Potansiyel Alanlar (APF) algoritmasını kullanmaktadır. Ayrıca sensör füzyonu ve lokalizasyon işlemleri için Genişletilmiş Kalman Filtresi (EKF) entegre edilmiştir.

## Kurulum ve Gereksinimler
Bu projenin çalıştırılabilmesi için bilgisayarınızda Python 3.x ve aşağıdaki kütüphanelerin yüklü olması gerekmektedir:
* `numpy`
* `matplotlib`

Gerekli kütüphaneleri kurmak için terminal veya komut istemcisine (CMD) aşağıdaki komutu yazabilirsiniz:
```bash
pip install numpy matplotlib
