import numpy as np
import matplotlib.pyplot as plt
from adim1_cevre_tasarimi import SimulasyonOrtami

class SensorSistemi:
    """
    Otonom mobil robot için Sensör Simülasyonu ve Veri İşleme modülü.
    LiDAR, IMU ve Tekerlek Enkoderi sensörlerini ve bu sensörlerin gürültülü (Gaussian noise)
    modellerini içerir. Ayrıca LiDAR verisi için eşikleme ve kümeleme (clustering) yapar.
    """
    def __init__(self, ortam, maksimum_lidar_menzili=10.0, lidar_isin_sayisi=72):
        self.ortam = ortam
        self.maksimum_lidar_menzili = maksimum_lidar_menzili
        self.lidar_isin_sayisi = lidar_isin_sayisi # 72 ışın = her 5 derecede 1 ışın
        
        # Gürültü (Noise) Standart Sapmaları (Gaussian Noise parametreleri)
        self.lidar_gurultu_std = 0.15      # Metre
        self.imu_aci_gurultu_std = 0.05    # Radyan
        self.imu_acisal_hiz_std = 0.02     # Radyan/saniye
        self.enkoder_hiz_gurultu_std = 0.1 # Metre/saniye
        
    def tekerlek_enkoderi_oku(self, gercek_v, gercek_omega):
        """
        Tekerlek enkoderi simülasyonu. 
        Gerçek çizgisel ve açısal hızlara Gauss gürültüsü ekleyerek okunan değerleri döner.
        """
        okunan_v = gercek_v + np.random.normal(0, self.enkoder_hiz_gurultu_std)
        # Diferansiyel sürüşte omega da tekerlek hızlarından elde edilir, gürültü ekleyelim.
        okunan_omega = gercek_omega + np.random.normal(0, self.enkoder_hiz_gurultu_std / 0.5) 
        return okunan_v, okunan_omega

    def imu_oku(self, gercek_teta, gercek_omega):
        """
        IMU (Inertial Measurement Unit) simülasyonu.
        Yönelim (Heading/Teta) ve açısal hız (Omega) ölçümlerine Gauss gürültüsü ekler.
        """
        okunan_teta = gercek_teta + np.random.normal(0, self.imu_aci_gurultu_std)
        okunan_omega = gercek_omega + np.random.normal(0, self.imu_acisal_hiz_std)
        return okunan_teta, okunan_omega

    def lidar_taramasi_yap(self, robot_x, robot_y, robot_teta):
        """
        2B ortamdaki engellere olan mesafeleri Işın İzleme (Raycasting) ile hesaplar
        ve üzerine Gauss gürültüsü ekleyerek ham LiDAR verisini oluşturur.
        """
        acilar = np.linspace(0, 2 * np.pi, self.lidar_isin_sayisi, endpoint=False)
        gercek_mesafeler = np.ones(self.lidar_isin_sayisi) * self.maksimum_lidar_menzili
        
        for i, aci in enumerate(acilar):
            mutlak_aci = robot_teta + aci
            # Işın boyunca adım adım ilerleme (Raymarching)
            for r in np.arange(0, self.maksimum_lidar_menzili, 0.1):
                px = robot_x + r * np.cos(mutlak_aci)
                py = robot_y + r * np.sin(mutlak_aci)
                
                carpisma = False
                # Çevre ortamındaki engellerle çarpışma kontrolü
                for (cx, cy, w, h) in self.ortam.engeller:
                    if abs(px - cx) <= w/2 and abs(py - cy) <= h/2:
                        gercek_mesafeler[i] = r
                        carpisma = True
                        break
                if carpisma:
                    break
                    
        # Gürültü Ekleme: Ham LiDAR verisine Gauss gürültüsü eklenmesi
        gurultu = np.random.normal(0, self.lidar_gurultu_std, self.lidar_isin_sayisi)
        ham_lidar_verisi = gercek_mesafeler + gurultu
        
        # Fiziksel sınırları aşmaması için veriyi kırp (0 ile maksimum menzil arasına)
        ham_lidar_verisi = np.clip(ham_lidar_verisi, 0.0, self.maksimum_lidar_menzili)
        
        return ham_lidar_verisi, acilar

    def lidar_veri_isleme(self, ham_mesafeler, acilar, esik_mesafesi=4.0, kume_mesafe_farki=0.6):
        """
        LiDAR verisine mesafe eşiklemesi (thresholding) uygular ve 
        yakın noktaları gruplayarak engelleri tespit eder (clustering).
        """
        esiklenmis_noktalar = []
        
        # 1. Mesafe Eşiklemesi (Thresholding)
        for i in range(len(ham_mesafeler)):
            if ham_mesafeler[i] < esik_mesafesi:
                esiklenmis_noktalar.append((i, ham_mesafeler[i], acilar[i]))
                
        # 2. Kümeleme (Clustering)
        kumeler = []
        guncel_kume = []
        
        for i in range(len(esiklenmis_noktalar)):
            idx, mesafe, aci = esiklenmis_noktalar[i]
            
            if not guncel_kume:
                guncel_kume.append(esiklenmis_noktalar[i])
            else:
                onceki_mesafe = guncel_kume[-1][1]
                onceki_idx = guncel_kume[-1][0]
                
                # Hem ardışık ışınlar olmalı hem de mesafeleri birbirine yakın olmalı
                if (idx - onceki_idx <= 2) and (abs(mesafe - onceki_mesafe) < kume_mesafe_farki):
                    guncel_kume.append(esiklenmis_noktalar[i])
                else:
                    kumeler.append(guncel_kume)
                    guncel_kume = [esiklenmis_noktalar[i]]
                    
        if guncel_kume:
            kumeler.append(guncel_kume)
            
        # 360 derece olduğu için son kume ile ilk kümeyi birleştirme kontrolü (Wrap-around)
        if len(kumeler) > 1:
            ilk_idx = kumeler[0][0][0]
            son_idx = kumeler[-1][-1][0]
            ilk_mesafe = kumeler[0][0][1]
            son_mesafe = kumeler[-1][-1][1]
            
            if (self.lidar_isin_sayisi - son_idx + ilk_idx <= 2) and (abs(ilk_mesafe - son_mesafe) < kume_mesafe_farki):
                kumeler[0] = kumeler[-1] + kumeler[0]
                kumeler.pop()
                
        return kumeler

def sensor_testini_calistir():
    """Sensör sınıflarını test etmek ve verileri görselleştirmek için fonksiyon."""
    ortam = SimulasyonOrtami()
    sensorler = SensorSistemi(ortam)
    
    # Robotun test konumu (Haritanın ortalarına doğru, engellere yakın bir yer)
    test_x, test_y, test_teta = 10.0, 6.0, np.pi/4
    gercek_v, gercek_w = 1.0, 0.5
    
    # Sensör okumaları (Gürültülü)
    enk_v, enk_w = sensorler.tekerlek_enkoderi_oku(gercek_v, gercek_w)
    imu_teta, imu_w = sensorler.imu_oku(test_teta, gercek_w)
    lidar_mesafeler, lidar_acilar = sensorler.lidar_taramasi_yap(test_x, test_y, test_teta)
    
    # LiDAR Veri İşleme
    kumeler = sensorler.lidar_veri_isleme(lidar_mesafeler, lidar_acilar, esik_mesafesi=6.0)
    
    print("--- ADIM 3: SENSÖR SİMÜLASYONU VE VERİ İŞLEME ---")
    print(f"Gerçek Hız (v, w): ({gercek_v:.2f}, {gercek_w:.2f})")
    print(f"Enkoder Okuması  : ({enk_v:.2f}, {enk_w:.2f})")
    print(f"Gerçek Yönelim   : {np.degrees(test_teta):.2f}°")
    print(f"IMU Okuması      : {np.degrees(imu_teta):.2f}°, w: {imu_w:.2f}")
    print(f"Tespit Edilen Engel Kümesi Sayısı: {len(kumeler)}")
    
    # Görselleştirme
    fig = plt.figure(figsize=(14, 6))
    
    # Sol Grafik: Harita ve Engeller
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(0, ortam.genislik)
    ax1.set_ylim(0, ortam.yukseklik)
    for (cx, cy, w, h) in ortam.engeller:
        ax1.add_patch(plt.Rectangle((cx-w/2, cy-h/2), w, h, color='gray', alpha=0.5))
    
    # Robotu çiz
    ax1.plot(test_x, test_y, 'bo', markersize=8, label='Robot Konumu')
    ax1.arrow(test_x, test_y, np.cos(test_teta), np.sin(test_teta), head_width=0.4, color='blue')
    
    # LiDAR ışınlarını çiz (Ham Veri)
    for i, aci in enumerate(lidar_acilar):
        mutlak_aci = test_teta + aci
        hx = test_x + lidar_mesafeler[i] * np.cos(mutlak_aci)
        hy = test_y + lidar_mesafeler[i] * np.sin(mutlak_aci)
        ax1.plot([test_x, hx], [test_y, hy], 'r-', alpha=0.1) # Kırmızı soluk çizgiler
        
    ax1.set_title("Simülasyon Ortamında LiDAR Taraması")
    ax1.grid(True, linestyle=':')
    
    # Sağ Grafik: İşlenmiş LiDAR Verisi (Polar Grafik Mantığı)
    ax2 = fig.add_subplot(122)
    acilar_derece = np.degrees(lidar_acilar)
    
    # Ham veriyi çiz
    ax2.plot(acilar_derece, lidar_mesafeler, 'o-', color='lightgray', label='Ham Veri (Gürültülü)', markersize=3)
    
    # Kümeleri (Clustering) çiz
    renkler = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
    for k_idx, kume in enumerate(kumeler):
        kume_acilari = [np.degrees(nokta[2]) for nokta in kume]
        kume_mesafeleri = [nokta[1] for nokta in kume]
        ax2.plot(kume_acilari, kume_mesafeleri, 'o', color=renkler[k_idx % len(renkler)], 
                 label=f'Tespit: Engel {k_idx+1}')
                 
    ax2.axhline(y=6.0, color='r', linestyle='--', label='Eşik Mesafesi (Threshold)')
    ax2.set_xlabel("LiDAR Işın Açısı (Derece)")
    ax2.set_ylabel("Ölçülen Mesafe (m)")
    ax2.set_title("LiDAR Veri İşleme (Eşikleme ve Kümeleme)")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sensor_testini_calistir()
