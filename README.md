# Sensör Füzyonu ve Lokalizasyon Kullanarak LiDAR Tabanlı Otonom Navigasyon

Bu proje, Bursa Teknik Üniversitesi Mekatronik Mühendisliği bölümü "Mobil Robotlar" dersi kapsamında Oğuzhan Akagündüz tarafından geliştirilmiştir.

## Proje Hakkında
Bu çalışmada, 2B düzlemde modellenmiş karmaşık bir ortamda hareket eden mobil bir robot için, **önemli bir paket teslimatını gerçekleştirmek üzere** otonom navigasyon ve lokalizasyon algoritmaları geliştirilmiştir. Robot, "Non-holonomic" kinematik modele sahip olup, engellerden kaçınmak için teğetsel kaydırma kuvveti (swirl force) destekli Yapay Potansiyel Alanlar (APF) algoritmasını kullanmaktadır. Ayrıca sensör füzyonu ve lokalizasyon işlemleri için Genişletilmiş Kalman Filtresi (EKF) entegre edilmiştir.

## Kurulum ve Gereksinimler
Bu projenin çalıştırılabilmesi için bilgisayarınızda Python 3.x ve aşağıdaki kütüphanelerin yüklü olması gerekmektedir:
* `numpy`
* `matplotlib`

Gerekli kütüphaneleri kurmak için terminal veya komut istemcisine (CMD) aşağıdaki komutu yazabilirsiniz:
```bash
pip install numpy matplotlib

Kodun Çalıştırılması
Bu depoyu bilgisayarınıza indirin veya otonom_navigasyon.py dosyasını kopyalayın.

Dosyanın bulunduğu dizinde terminali açın.

Aşağıdaki komut ile simülasyonu başlatın:

Bash
python otonom_navigasyon.py
Kod çalıştırıldığında; otonom navigasyon haritası, hata analizi (RMSE) ve LiDAR sensör verilerini gösteren 3 farklı grafik ekranınıza gelecektir.

Yapay Zeka Kullanım Beyanı
Kullanılan Araçlar: Gemini (1.5 Pro) ve ChatGPT (GPT-4o).

Kullanım Alanları: Kalman Filtresi ve Navigasyon algoritmasının Python kodlarının oluşturulması, kod hatalarının ayıklanması ve raporun akademik dil düzenlemesi.

Öğrenci Katkısı: Senaryonun tasarımı, engellerin fiziksel koordinatlarının belirlenmesi, navigasyon parametrelerinin optimizasyonu ve simülasyon testleri.

Açıklama: Yapay zeka araçları yardımcı araç olarak kullanılmıştır. Nihai kod, senaryo, deney sonuçları ve rapor değerlendirmeleri öğrenci tarafından kontrol edilerek teslim edilmiştir.
