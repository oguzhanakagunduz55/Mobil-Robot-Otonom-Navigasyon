# Sensör Füzyonu ve Lokalizasyon Kullanarak LiDAR Tabanlı Otonom Ev Robot Süpürgesi

Bu proje, Bursa Teknik Üniversitesi Mekatronik Mühendisliği bölümü **Mobil Robotlar** dersi kapsamında geliştirilmiştir.

## Proje Hakkında

Bu çalışmada, LiDAR tabanlı otonom navigasyon gerçekleştiren bir mobil robot sistemi tasarlanmıştır. Sistem; haritalama, lokalizasyon, sensör füzyonu ve engelden kaçınma algoritmalarını bir araya getirerek gerçekçi bir ev ortamında görev yapmaktadır.

Simülasyon ortamı 30x15 metre boyutlarında bir ev senaryosundan oluşmaktadır. Ortam içerisinde 10 adet statik engel (mobilya) ve 1 adet dinamik engel (hareketli kedi) bulunmaktadır.

Proje iki ana aşamadan oluşmaktadır:

---

# 1. Drone ile Statik SLAM Haritalama

İlk aşamada tavan hizasında hareket eden keşif dronu, ortamı LiDAR sensörü ile taramaktadır.

Drone:
- zikzak hareket rotaları oluşturarak tüm evi tarar,
- LiDAR verilerini toplar,
- Occupancy Grid Mapping yöntemiyle harita oluşturur,
- hücre oylama (Occupancy Voting) filtresi kullanarak gürültülü ölçümleri temizler.

Bu sayede ortamın daha kararlı ve pürüzsüz bir 2B haritası elde edilir.

---

# 2. Robot Süpürge ile Dinamik Otonom Navigasyon

İkinci aşamada robot süpürge, başlangıç noktasından hedefe otonom olarak ilerlemektedir.

Robot:
- LiDAR sensörü ile çevresini sürekli tarar,
- EKF (Extended Kalman Filter) kullanarak lokalizasyon yapar,
- sensör füzyonu sayesinde odometri hatalarını düzeltir,
- APF (Artificial Potential Field) algoritması ile engellerden kaçınır,
- dinamik engel olarak hareket eden kediyi algılayıp reaktif kaçış manevraları uygular.

Bu yapı sayesinde robot hedefe güvenli ve kararlı şekilde ulaşabilmektedir.

---

# Kullanılan Yöntemler

## Kinematik Model
- Differential Drive (Diferansiyel Sürüş)

## Sensör Füzyonu
- Odometri + LiDAR entegrasyonu
- Extended Kalman Filter (EKF)

## Haritalama
- Occupancy Grid Mapping
- Occupancy Voting Filter

## Navigasyon
- Artificial Potential Field (APF)
- Reaktif Engelden Kaçınma
- Waypoint Tabanlı Yol Planlama

---

# Simülasyon Özellikleri

- Gerçek zamanlı LiDAR tarama animasyonu
- Hareketli robot ve sensör görselleştirmesi
- Statik ve dinamik engeller
- Gerçek yol ve tahmini yol karşılaştırması
- EKF hata analizi
- Sensör gürültü modellemesi
- Çarpışma kontrol sistemi
- Canlı navigasyon animasyonu

---

# Kullanılan Teknolojiler

- Python 3
- NumPy
- Matplotlib
- FilterPy

---

# Kurulum

Projeyi çalıştırmadan önce aşağıdaki kütüphanelerin kurulması gerekmektedir:

```bash
pip install numpy matplotlib filterpy
```

---

# Çalıştırma

Terminal veya komut istemcisinden proje klasörüne giderek aşağıdaki komutu çalıştırın:

```bash
python adim8_animasyon.py
```

Simülasyon başlatıldıktan sonra:

- robotun gerçek zamanlı hareketi,
- LiDAR taramaları,
- engelden kaçınma davranışları,
- EKF lokalizasyon sonuçları

ekranda canlı olarak görüntülenmektedir.

Simülasyon tamamlandığında otomatik olarak analiz sonuçları oluşturulmaktadır.

---

# Proje Çıktıları

Proje kapsamında:

- SLAM tabanlı haritalama,
- LiDAR sensör simülasyonu,
- sensör füzyonu,
- EKF lokalizasyonu,
- otonom navigasyon,
- engelden kaçınma,
- gerçek zamanlı animasyon

başarıyla gerçekleştirilmiştir.

---

# Yapay Zeka Kullanım Beyanı

## Kullanılan Araçlar
- ChatGPT
- Gemini

## Kullanım Amaçları
- Python kod taslaklarının oluşturulması
- EKF algoritması geliştirme
- LiDAR veri işleme
- APF navigasyon algoritması
- Simülasyon hata ayıklama (debugging)
- Dokümantasyon düzenleme

## Öğrenci Katkısı
- Proje senaryosunun oluşturulması
- Ev ortamı tasarımı
- Navigasyon yapısının geliştirilmesi
- Parametre optimizasyonları
- Test süreçleri
- Simülasyon analizleri
- Son düzenlemeler

tamamen proje geliştiricisi tarafından gerçekleştirilmiştir.

---

# Geliştirici

Oğuzhan Akagündüz  
Bursa Teknik Üniversitesi 
Mekatronik Mühendisliği

---
