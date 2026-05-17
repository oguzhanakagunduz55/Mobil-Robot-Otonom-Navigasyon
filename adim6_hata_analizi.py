import numpy as np
import matplotlib.pyplot as plt

# Adım 4'te yazdığımız Genişletilmiş Kalman Filtresi (EKF) modülünü içe aktarıyoruz
from adim4_lokalizasyon_ekf import GenisletilmisKalmanFiltresi

def hata_analizi_ve_dogrulama():
    """
    Adım 6: Simülasyon bittikten sonra sistemin ne kadar iyi çalıştığını ölçmek için 
    Hata Analizi (RMSE ve MAE) yapar ve sonuçları görselleştirir.
    """
    dt = 0.1
    toplam_zaman = 40.0
    adim_sayisi = int(toplam_zaman / dt)
    
    baslangic_x, baslangic_y, baslangic_teta = 0.0, 0.0, 0.0
    
    # Gerçek (Ground Truth), Dead Reckoning (DR) ve EKF (Kalman) durumları
    gercek_x, gercek_y, gercek_teta = baslangic_x, baslangic_y, baslangic_teta
    dr_x, dr_y, dr_teta = baslangic_x, baslangic_y, baslangic_teta
    
    ekf = GenisletilmisKalmanFiltresi(baslangic_x, baslangic_y, baslangic_teta, dt)
    
    kayit_gercek = {'x': [gercek_x], 'y': [gercek_y]}
    kayit_dr = {'x': [dr_x], 'y': [dr_y]}
    kayit_ekf = {'x': [baslangic_x], 'y': [baslangic_y]}
    
    # Gürültü Parametreleri (Dead Reckoning'in ne kadar sapacağını belirler)
    enkoder_hata_v = 0.25      # m/s
    enkoder_hata_w = np.radians(12.0) # rad/s
    gps_hata_xy = 1.0          # metre (Yüksek GPS gürültüsü)
    imu_hata_teta = np.radians(8.0)  # radyan
    
    # Hata Metrikleri için Diziler (Zamana bağlı hataları depolamak için)
    dr_hatalari = []
    ekf_hatalari = []
    zaman_adimlari = []
    
    print("Simülasyon çalıştırılıyor. Gürültülü sensörlerle veri toplanıyor...")
    
    for i in range(adim_sayisi):
        zaman = i * dt
        zaman_adimlari.append(zaman)
        
        # Test için karmaşık bir sekiz (8) ve virajlı rota komutları
        v_komut = 1.2 + 0.3 * np.sin(zaman * 0.2)
        w_komut = 0.6 * np.sin(zaman * 0.3)
        
        # 1. GERÇEK DÜNYA (Ground Truth)
        gercek_x += v_komut * np.cos(gercek_teta) * dt
        gercek_y += v_komut * np.sin(gercek_teta) * dt
        gercek_teta += w_komut * dt
        gercek_teta = np.arctan2(np.sin(gercek_teta), np.cos(gercek_teta))
        
        # 2. SENSÖR GÜRÜLTÜLERİ
        v_okunan = v_komut + np.random.normal(0, enkoder_hata_v)
        w_okunan = w_komut + np.random.normal(0, enkoder_hata_w)
        
        z_x = gercek_x + np.random.normal(0, gps_hata_xy)
        z_y = gercek_y + np.random.normal(0, gps_hata_xy)
        z_teta = gercek_teta + np.random.normal(0, imu_hata_teta)
        
        # 3. DEAD RECKONING (Sadece Odometri - Hata Kümülatif Büyür)
        dr_x += v_okunan * np.cos(dr_teta) * dt
        dr_y += v_okunan * np.sin(dr_teta) * dt
        dr_teta += w_okunan * dt
        dr_teta = np.arctan2(np.sin(dr_teta), np.cos(dr_teta))
        
        # 4. EKF GÜNCELLEMESİ (Sensör Füzyonu)
        ekf.tahmin_et(v_okunan, w_okunan)
        # GPS ölçümü 10 adımda bir (1 saniyede bir) gelsin. Gerçekçi kesintili bağlantı senaryosu.
        if i % 10 == 0:
            durum_est = ekf.guncelle([z_x, z_y, z_teta])
        else:
            durum_est = ekf.x_est
            
        ekf_x, ekf_y = durum_est[0, 0], durum_est[1, 0]
        
        # Kayıtlar
        kayit_gercek['x'].append(gercek_x); kayit_gercek['y'].append(gercek_y)
        kayit_dr['x'].append(dr_x); kayit_dr['y'].append(dr_y)
        kayit_ekf['x'].append(ekf_x); kayit_ekf['y'].append(ekf_y)
        
        # Anlık Hata Hesaplamaları (Öklid Mesafesi - L2 Norm)
        # Gerçek konum ile tahmin edilen konumlar arasındaki mesafe farkı
        anlik_hata_dr = np.hypot(gercek_x - dr_x, gercek_y - dr_y)
        anlik_hata_ekf = np.hypot(gercek_x - ekf_x, gercek_y - ekf_y)
        
        dr_hatalari.append(anlik_hata_dr)
        ekf_hatalari.append(anlik_hata_ekf)

    # =====================================================================
    # --- MATEMATİKSEL DOĞRULAMA VE HATA METRİKLERİ (RMSE VE MAE) ---
    # =====================================================================
    dr_hatalari = np.array(dr_hatalari)
    ekf_hatalari = np.array(ekf_hatalari)
    
    # 1. RMSE (Root Mean Square Error - Kök Ortalama Kare Hatası)
    # Büyük hataları daha çok cezalandırdığı için regresyon analizlerinde çok önemlidir.
    rmse_dr = np.sqrt(np.mean(dr_hatalari**2))
    rmse_ekf = np.sqrt(np.mean(ekf_hatalari**2))
    
    # 2. MAE (Mean Absolute Error - Ortalama Mutlak Hata)
    # Hataların mutlak değerlerinin ortalamasıdır. (Bütün hatalara eşit ağırlık verir)
    mae_dr = np.mean(np.abs(dr_hatalari))
    mae_ekf = np.mean(np.abs(ekf_hatalari))

    print("\n" + "="*60)
    print("ADIM 6: HATA ANALİZİ VE MATEMATİKSEL DOĞRULAMA SONUÇLARI")
    print("="*60)
    print(f"{'Metrik':<20} | {'Dead Reckoning (Odometri)':<25} | {'EKF (Sensör Füzyonu)'}")
    print("-" * 60)
    print(f"{'RMSE':<20} | {rmse_dr:.4f} m{'':<22} | {rmse_ekf:.4f} m")
    print(f"{'MAE':<20} | {mae_dr:.4f} m{'':<22} | {mae_ekf:.4f} m")
    print(f"{'Maksimum Anlık Hata':<20} | {np.max(dr_hatalari):.4f} m{'':<22} | {np.max(ekf_hatalari):.4f} m")
    print("="*60)
    print(f"ÖZET: EKF kullanımı RMSE'yi {(rmse_dr/rmse_ekf):.1f} kat, MAE'yi ise {(mae_dr/mae_ekf):.1f} kat iyileştirmiştir.")
    print("="*60)

    # Görselleştirme (2 Alt Grafik)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Adım 6: Lokalizasyon Hata Analizi ve Metrikler", fontsize=16, fontweight='bold', y=0.98)
    
    # --- Sol Grafik: Yörünge Karşılaştırması ---
    ax1.plot(kayit_dr['x'], kayit_dr['y'], 'r--', linewidth=1.5, alpha=0.7, label='Dead Reckoning (Hata Birikir)')
    ax1.plot(kayit_gercek['x'], kayit_gercek['y'], 'g-', linewidth=3, label='Gerçek Konum (Ground Truth)')
    ax1.plot(kayit_ekf['x'], kayit_ekf['y'], 'b-', linewidth=2, label='EKF Tahmini (Düzeltilmiş)')
    ax1.set_title("Robotun İzlediği Yörüngeler")
    ax1.set_xlabel("X Koordinatı (m)")
    ax1.set_ylabel("Y Koordinatı (m)")
    ax1.legend()
    ax1.grid(True, linestyle=':')
    ax1.axis('equal')
    
    # --- Sağ Grafik: Zamanla Hata Analizi (RMSE ve MAE Gösterimi) ---
    ax2.plot(zaman_adimlari, dr_hatalari, 'r-', alpha=0.6, label='Dead Reckoning Anlık Hatası')
    ax2.plot(zaman_adimlari, ekf_hatalari, 'b-', linewidth=2, label='EKF Anlık Hatası')
    
    # Metrik çizgileri
    ax2.axhline(y=rmse_ekf, color='cyan', linestyle='--', linewidth=2, label=f'EKF RMSE ({rmse_ekf:.2f}m)')
    ax2.axhline(y=mae_ekf, color='navy', linestyle=':', linewidth=2, label=f'EKF MAE ({mae_ekf:.2f}m)')
    
    ax2.set_title("Zaman İçerisinde Konum Hatası Değişimi (Öklid Mesafesi)")
    ax2.set_xlabel("Zaman (s)")
    ax2.set_ylabel("Konum Sapması (m)")
    ax2.legend()
    ax2.grid(True, linestyle=':')
    
    # Sağ üst köşeye text box ile analiz sonuçları
    textstr = '\n'.join((
        f'Hata Metrikleri (L2 Norm):',
        f'DR RMSE : {rmse_dr:.2f}m',
        f'EKF RMSE: {rmse_ekf:.2f}m',
        f'DR MAE  : {mae_dr:.2f}m',
        f'EKF MAE : {mae_ekf:.2f}m'
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
             
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Suptitle ile çakışmaması için
    plt.show()

if __name__ == "__main__":
    hata_analizi_ve_dogrulama()
