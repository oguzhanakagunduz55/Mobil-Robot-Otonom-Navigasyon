import numpy as np
import matplotlib.pyplot as plt

class GenisletilmisKalmanFiltresi:
    """
    Otonom Robot Lokalizasyonu için Genişletilmiş Kalman Filtresi (EKF).
    Non-linear kinematik modeli doğrusallaştırarak sensör füzyonu yapar.
    """
    def __init__(self, baslangic_x, baslangic_y, baslangic_teta, dt=0.1):
        # Durum Vektörü x_est = [x, y, theta]^T
        self.x_est = np.array([[baslangic_x], [baslangic_y], [baslangic_teta]], dtype=float)
        self.dt = dt
        
        # Hata Kovaryans Matrisi (P) - Başlangıç belirsizliği
        self.P = np.eye(3) * 0.1
        
        # Süreç (Process) Gürültüsü Kovaryansı (Q) 
        # (Enkoderden gelen hatalar: x, y, theta varyansları)
        self.Q = np.diag([0.01, 0.01, np.radians(1.0)]) ** 2
        
        # Ölçüm (Measurement) Gürültüsü Kovaryansı (R) 
        # (Örn: GPS/LiDAR Eşleşmesi ve IMU hataları)
        self.R = np.diag([0.5, 0.5, np.radians(5.0)]) ** 2

    def tahmin_et(self, v, omega):
        """
        EKF TAHMİN (PREDICTION) ADIMI: 
        Sadece Tekerlek Enkoderinden gelen (v, omega) hız komutları ile konum tahmini yapar.
        """
        teta = self.x_est[2, 0]
        
        # 1. Durum Tahmini (Kinematik Model)
        self.x_est[0, 0] += v * np.cos(teta) * self.dt
        self.x_est[1, 0] += v * np.sin(teta) * self.dt
        self.x_est[2, 0] += omega * self.dt
        self.x_est[2, 0] = np.arctan2(np.sin(self.x_est[2, 0]), np.cos(self.x_est[2, 0]))
        
        # 2. Jacobian Matrisi (F) - Durum Geçiş Fonksiyonunun Kısmi Türevi
        # Non-linear olan modeli o anki duruma göre doğrusallaştırır.
        F = np.array([
            [1.0, 0.0, -v * np.sin(teta) * self.dt],
            [0.0, 1.0,  v * np.cos(teta) * self.dt],
            [0.0, 0.0, 1.0]
        ])
        
        # 3. Kovaryans Tahmini (Hata matrisinin büyümesi)
        self.P = F @ self.P @ F.T + self.Q
        
        return self.x_est.copy()

    def guncelle(self, z):
        """
        EKF GÜNCELLEME (CORRECTION) ADIMI:
        Mutlak ölçüm sensörlerinden (GPS + IMU) gelen z = [x, y, teta] verisi ile tahmini düzeltir.
        """
        z = np.array(z).reshape(3, 1)
        
        # Ölçüm Matrisi (H) - Durumu ölçüme dönüştürür (Burada birebir aynı olduğu için Birim Matris)
        H = np.eye(3)
        
        # Yenilik (Innovation) - Ölçülen ile Tahmin Edilen Arasındaki Fark
        y = z - (H @ self.x_est)
        y[2, 0] = np.arctan2(np.sin(y[2, 0]), np.cos(y[2, 0])) # Açısal farkı doğru aralıkta tut
        
        # Yenilik Kovaryansı (S)
        S = H @ self.P @ H.T + self.R
        
        # Kalman Kazancı (K) - Ölçüme mi yoksa Tahmine mi daha çok güveneceğiz?
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 1. Durumu Güncelle
        self.x_est = self.x_est + (K @ y)
        self.x_est[2, 0] = np.arctan2(np.sin(self.x_est[2, 0]), np.cos(self.x_est[2, 0]))
        
        # 2. Kovaryansı Güncelle (Belirsizlik azalır)
        self.P = (np.eye(3) - (K @ H)) @ self.P
        
        return self.x_est.copy()

def lokalizasyon_ve_sensor_fuzyonu_testi():
    """Tüm simülasyon boyunca 3 farklı konumu hesaplar, eşzamanlı kaydeder ve çizer."""
    dt = 0.1
    toplam_zaman = 25.0
    adim_sayisi = int(toplam_zaman / dt)
    
    baslangic_x, baslangic_y, baslangic_teta = 2.0, 2.0, 0.0
    
    # 1. GERÇEK DURUM (Ground Truth - İdeal Dünya)
    gercek_x, gercek_y, gercek_teta = baslangic_x, baslangic_y, baslangic_teta
    kayit_gercek = {'x': [gercek_x], 'y': [gercek_y]}
    
    # 2. DEAD RECKONING DURUMU (Sadece Enkoder, Hata birikir)
    dr_x, dr_y, dr_teta = baslangic_x, baslangic_y, baslangic_teta
    kayit_dr = {'x': [dr_x], 'y': [dr_y]}
    
    # 3. EKF DURUMU (Sensör Füzyonu)
    ekf = GenisletilmisKalmanFiltresi(baslangic_x, baslangic_y, baslangic_teta, dt)
    kayit_ekf = {'x': [baslangic_x], 'y': [baslangic_y]}
    
    # Gürültü Parametreleri (Standart Sapmalar)
    enkoder_hata_v = 0.15      # m/s
    enkoder_hata_w = np.radians(8.0) # rad/s
    gps_hata_xy = 0.6          # metre
    imu_hata_teta = np.radians(5.0)  # radyan
    
    print("Simülasyon başlatıldı. EKF, Dead Reckoning ve Gerçek konum hesaplanıyor...")
    
    for i in range(adim_sayisi):
        zaman = i * dt
        
        # Basit bir Rota Komutu (Düz git, dön, kıvrıl vs.)
        v_komut = 1.2
        w_komut = 0.4 * np.sin(zaman * 0.5) # Sinüsoidal bir dönüş
        
        # --- 1. GERÇEK DÜNYA HESAPLAMASI (Ground Truth) ---
        gercek_x += v_komut * np.cos(gercek_teta) * dt
        gercek_y += v_komut * np.sin(gercek_teta) * dt
        gercek_teta += w_komut * dt
        gercek_teta = np.arctan2(np.sin(gercek_teta), np.cos(gercek_teta))
        
        kayit_gercek['x'].append(gercek_x)
        kayit_gercek['y'].append(gercek_y)
        
        # --- 2. SENSÖR OKUMALARI (Gürültü Eklenmiş) ---
        v_okunan = v_komut + np.random.normal(0, enkoder_hata_v)
        w_okunan = w_komut + np.random.normal(0, enkoder_hata_w)
        
        z_x = gercek_x + np.random.normal(0, gps_hata_xy)
        z_y = gercek_y + np.random.normal(0, gps_hata_xy)
        z_teta = gercek_teta + np.random.normal(0, imu_hata_teta)
        
        # --- 3. DEAD RECKONING HESAPLAMASI (Odometri) ---
        dr_x += v_okunan * np.cos(dr_teta) * dt
        dr_y += v_okunan * np.sin(dr_teta) * dt
        dr_teta += w_okunan * dt
        dr_teta = np.arctan2(np.sin(dr_teta), np.cos(dr_teta))
        
        kayit_dr['x'].append(dr_x)
        kayit_dr['y'].append(dr_y)
        
        # --- 4. KALMAN FİLTRESİ HESAPLAMASI (EKF) ---
        # Adım A: Tahmin et (Sadece Enkoder ile)
        ekf.tahmin_et(v_okunan, w_okunan)
        
        # Adım B: Güncelle (GPS ve IMU ölçümleri ile)
        # Örn: Mutlak konum ölçümü her zaman değil, her 5 adımda (0.5 saniyede) bir gelsin
        if i % 5 == 0:
            durum_est = ekf.guncelle([z_x, z_y, z_teta])
        else:
            durum_est = ekf.x_est
            
        kayit_ekf['x'].append(durum_est[0, 0])
        kayit_ekf['y'].append(durum_est[1, 0])

    print("Simülasyon tamamlandı. Sonuçlar çizdiriliyor...")

    # Görselleştirme: 3 Farklı Konumun Karşılaştırması
    plt.figure(figsize=(12, 8))
    
    # Dead Reckoning (Giderek Hata Biriktiren Mavi Kesik Çizgi)
    plt.plot(kayit_dr['x'], kayit_dr['y'], 'r--', linewidth=2, alpha=0.7, label='Dead Reckoning (Sadece Enkoder, Hata Birikir)')
    
    # Gerçek Konum (Kusursuz İdeal Yol, Yeşil Kalın Çizgi)
    plt.plot(kayit_gercek['x'], kayit_gercek['y'], 'g-', linewidth=3, label='Gerçek Konum (Ground Truth)')
    
    # EKF Kestirimi (Sensör Füzyonu sonucu düzeltilmiş yol, Mavi Çizgi)
    plt.plot(kayit_ekf['x'], kayit_ekf['y'], 'b-', linewidth=2, label='EKF Tahmini (Sensör Füzyonu - Düzeltilmiş)')
    
    # Başlangıç ve Bitiş Noktaları
    plt.plot(kayit_gercek['x'][0], kayit_gercek['y'][0], 'ko', markersize=8, label='Başlangıç Noktası')
    plt.plot(kayit_gercek['x'][-1], kayit_gercek['y'][-1], 'go', markersize=8)
    plt.plot(kayit_ekf['x'][-1], kayit_ekf['y'][-1], 'bo', markersize=8)
    plt.plot(kayit_dr['x'][-1], kayit_dr['y'][-1], 'ro', markersize=8)

    plt.title("Adım 4: Sensör Füzyonu ve Lokalizasyon Karşılaştırması (EKF)", pad=15, fontsize=14, fontweight='bold')
    plt.xlabel("X Koordinatı (m)")
    plt.ylabel("Y Koordinatı (m)")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    
    # Hata metin kutusu ekleme (Son konumlar arasındaki mesafe farkı)
    hata_dr = np.sqrt((kayit_dr['x'][-1] - kayit_gercek['x'][-1])**2 + (kayit_dr['y'][-1] - kayit_gercek['y'][-1])**2)
    hata_ekf = np.sqrt((kayit_ekf['x'][-1] - kayit_gercek['x'][-1])**2 + (kayit_ekf['y'][-1] - kayit_gercek['y'][-1])**2)
    
    textstr = '\n'.join((
        f'Bitiş Noktası Hataları:',
        f'Dead Reckoning Hatası: {hata_dr:.2f}m',
        f'EKF Hatası: {hata_ekf:.2f}m'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.gca().text(0.65, 0.15, textstr, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    lokalizasyon_ve_sensor_fuzyonu_testi()
