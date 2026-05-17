import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Önceki modüllerden gerekli olanları alıyoruz
from adim1_cevre_tasarimi import SimulasyonOrtami
from adim3_sensor_simulasyonu import SensorSistemi
from adim4_lokalizasyon_ekf import GenisletilmisKalmanFiltresi

def rapor_icin_veri_uret():
    """Rapor grafikleri için gerekli tüm simülasyon verilerini tek bir senaryoda üretir."""
    ortam = SimulasyonOrtami()
    sensorler = SensorSistemi(ortam)
    
    dt = 0.1
    toplam_zaman = 30.0
    adim_sayisi = int(toplam_zaman / dt)
    
    # Başlangıç Durumları
    baslangic_x, baslangic_y, baslangic_teta = ortam.baslangic_noktasi[0], ortam.baslangic_noktasi[1], np.pi/4
    g_x, g_y, g_teta = baslangic_x, baslangic_y, baslangic_teta
    dr_x, dr_y, dr_teta = baslangic_x, baslangic_y, baslangic_teta
    
    ekf = GenisletilmisKalmanFiltresi(baslangic_x, baslangic_y, baslangic_teta, dt)
    
    # Ara Hedefler (Global Planner Waypoints) - Yeni Başlangıç Noktasına Göre Uyarlanmış
    ara_hedefler = [(7.0, 6.0), (12.0, 7.0), (17.0, 7.0), (22.0, 11.0), ortam.hedef_nokta]
    aktif_hedef_idx = 0
    hedef_kabul_yaricapi = 0.1
    
    # Kayıt Dizileri
    zaman_dizisi = []
    kayit_gercek = {'x': [], 'y': [], 'teta': []}
    kayit_ekf = {'x': [], 'y': [], 'teta': []}
    kayit_dr = {'x': [], 'y': [], 'teta': []}
    hata_dizisi = []
    
    # Örnek bir anın LiDAR verisi için (Simülasyonun ortalarında kaydedilecek)
    ornek_lidar_ham = None
    ornek_lidar_kume = None
    ornek_lidar_acilar = None

    print("Rapor verileri simüle ediliyor...")
    
    for i in range(adim_sayisi):
        zaman = i * dt
        zaman_dizisi.append(zaman)
        # APF (Yapay Potansiyel Alan) ile engellerden kaçınan hız komutlarını hesaplama
        hedef_x, hedef_y = ara_hedefler[aktif_hedef_idx]
        
        # Çekici Kuvvet
        f_x = 1.5 * (hedef_x - g_x)
        f_y = 1.5 * (hedef_y - g_y)
        mesafe = np.hypot(hedef_x - g_x, hedef_y - g_y)
        if mesafe > 1.0:
            f_x = (f_x / mesafe) * 2.0
            f_y = (f_y / mesafe) * 2.0
            
        # İtici Kuvvet (Anlık Lidar'dan)
        anlik_lidar_ham, anlik_lidar_acilar = sensorler.lidar_taramasi_yap(g_x, g_y, g_teta)
        for j in range(len(anlik_lidar_ham)):
            d = anlik_lidar_ham[j]
            if 0.1 < d < 3.0:
                mutlak_aci = g_teta + anlik_lidar_acilar[j]
                kuvvet = 3.0 * (1.0 / d - 1.0 / 3.0) * (1.0 / (d**2))
                f_x += kuvvet * np.cos(mutlak_aci + np.pi)
                f_y += kuvvet * np.sin(mutlak_aci + np.pi)
                
        # Duvar İtmeleri
        if g_x < 1.0: f_x += 3.0 * (1.0/max(g_x, 0.1))**2
        if g_x > ortam.genislik - 1.0: f_x -= 3.0 * (1.0/max(ortam.genislik - g_x, 0.1))**2
        if g_y < 1.0: f_y += 3.0 * (1.0/max(g_y, 0.1))**2
        if g_y > ortam.yukseklik - 1.0: f_y -= 3.0 * (1.0/max(ortam.yukseklik - g_y, 0.1))**2
        
        istenen_aci = np.arctan2(f_y, f_x)
        aci_farki = istenen_aci - g_teta
        aci_farki = np.arctan2(np.sin(aci_farki), np.cos(aci_farki))
        
        if abs(aci_farki) > np.pi / 3:
            v_komut = 0.2
            w_komut = np.sign(aci_farki) * np.pi
        else:
            v_komut = 2.0 * (1.0 - abs(aci_farki)/np.pi)
            w_komut = 3.0 * aci_farki
            
        # Hedefe varıldı mı kontrol et
        if mesafe < hedef_kabul_yaricapi and aktif_hedef_idx < len(ara_hedefler) - 1:
            aktif_hedef_idx += 1
        
        # 1. Gerçek Hareket
        g_x += v_komut * np.cos(g_teta) * dt
        g_y += v_komut * np.sin(g_teta) * dt
        g_teta += w_komut * dt
        g_teta = np.arctan2(np.sin(g_teta), np.cos(g_teta))
        
        kayit_gercek['x'].append(g_x); kayit_gercek['y'].append(g_y); kayit_gercek['teta'].append(g_teta)
        
        # 2. Sensör Okumaları ve EKF
        v_olculen, w_olculen = sensorler.tekerlek_enkoderi_oku(v_komut, w_komut)
        
        # Dead Reckoning (Sadece Enkoder)
        dr_x += v_olculen * np.cos(dr_teta) * dt
        dr_y += v_olculen * np.sin(dr_teta) * dt
        dr_teta += w_olculen * dt
        dr_teta = np.arctan2(np.sin(dr_teta), np.cos(dr_teta))
        kayit_dr['x'].append(dr_x); kayit_dr['y'].append(dr_y); kayit_dr['teta'].append(dr_teta)
        
        ekf.tahmin_et(v_olculen, w_olculen)
        
        # Her 10 adımda bir GPS (Mutlak ölçüm) gelsin
        if i % 10 == 0:
            z_x = g_x + np.random.normal(0, 0.5)
            z_y = g_y + np.random.normal(0, 0.5)
            z_teta = g_teta + np.random.normal(0, np.radians(5))
            durum_est = ekf.guncelle([z_x, z_y, z_teta])
        else:
            durum_est = ekf.x_est
            
        e_x, e_y, e_teta = durum_est[0, 0], durum_est[1, 0], durum_est[2, 0]
        kayit_ekf['x'].append(e_x); kayit_ekf['y'].append(e_y); kayit_ekf['teta'].append(e_teta)
        
        # Hata
        hata_dizisi.append(np.hypot(g_x - e_x, g_y - e_y))
        
        # LiDAR verisini tam 15. saniyede alıp kaydedelim
        if abs(zaman - 15.0) < dt/2:
            ornek_lidar_ham, ornek_lidar_acilar = sensorler.lidar_taramasi_yap(g_x, g_y, g_teta)
            ornek_lidar_kume = sensorler.lidar_veri_isleme(ornek_lidar_ham, ornek_lidar_acilar, esik_mesafesi=6.0)

    return ortam, ara_hedefler, zaman_dizisi, kayit_gercek, kayit_ekf, kayit_dr, hata_dizisi, ornek_lidar_ham, ornek_lidar_kume, ornek_lidar_acilar

def grafik_raporlarini_olustur():
    """Belirtilen 5 farklı zorunlu grafiği Matplotlib ile çizer."""
    (ortam, ara_hedefler, zaman_dizisi, kayit_gercek, kayit_ekf, kayit_dr, hata_dizisi, 
     lidar_ham, lidar_kume, lidar_acilar) = rapor_icin_veri_uret()

    # Ortak Figure Ayarları (Büyük Rapor Panosu)
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("ADIM 7: MOBİL ROBOT SİMÜLASYONU GÖRSELLEŞTİRME VE RAPORLAMA", fontsize=18, fontweight='bold', y=0.98)
    
    # Kılavuz (Grid) yapısı: 2 satır 3 sütun = 6 hücre
    # 1. Ortam Haritası (Satır 0, Sütun 0)
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_xlim(0, ortam.genislik); ax1.set_ylim(0, ortam.yukseklik)
    for (cx, cy, w, h) in ortam.engeller:
        ax1.add_patch(patches.Rectangle((cx-w/2, cy-h/2), w, h, color='#808080', alpha=0.8, label='Engel' if cx==ortam.engeller[0][0] else ""))
    ax1.plot(ortam.baslangic_noktasi[0], ortam.baslangic_noktasi[1], 'go', markersize=10, label='Başlangıç Noktası')
    ax1.plot(ortam.hedef_nokta[0], ortam.hedef_nokta[1], 'ro', markersize=10, label='Hedef Nokta')
    ax1.set_title("1. Ortam Haritası\n(Engeller ve Sınırlar)", fontweight='bold')
    ax1.set_xlabel("X Konumu (Metre)")
    ax1.set_ylabel("Y Konumu (Metre)")
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, linestyle=':')

    # 2. Yol Grafiği (Satır 0, Sütun 1 ve 2'yi birleştir)
    ax2 = plt.subplot(2, 3, (2, 3))
    ax2.set_xlim(0, ortam.genislik); ax2.set_ylim(0, ortam.yukseklik)
    for (cx, cy, w, h) in ortam.engeller:
        ax2.add_patch(patches.Rectangle((cx-w/2, cy-h/2), w, h, color='#A0A0A0', alpha=0.5))
    
    # Global Waypoints (Planlanan Yol) - İsteğe bağlı olarak gizlendi
    # wx = [ortam.baslangic_noktasi[0]] + [h[0] for h in ara_hedefler]
    # wy = [ortam.baslangic_noktasi[1]] + [h[1] for h in ara_hedefler]
    # ax2.plot(wx, wy, color='orange', linestyle='--', linewidth=2, marker='X', markersize=8, label='Planlanan Yol (Global Planner)')
    
    # İzlenen Yol (EKF Kestirimi, Gerçek Yol, Dead Reckoning)
    ax2.plot(kayit_dr['x'], kayit_dr['y'], 'r--', linewidth=2, alpha=0.7, label='Dead Reckoning (Sadece Odometri)')
    ax2.plot(kayit_gercek['x'], kayit_gercek['y'], 'g-', linewidth=4, alpha=0.6, label='Gerçek Yol (Ground Truth)')
    ax2.plot(kayit_ekf['x'], kayit_ekf['y'], 'b-', linewidth=2, label='EKF Kestirimi (Sensör Füzyonu)')
    
    ax2.plot(ortam.hedef_nokta[0], ortam.hedef_nokta[1], 'ro', markersize=12, label='Bitiş Noktası')
    
    ax2.set_title("2. Yol Grafiği\n(Gerçek Yol vs EKF vs Dead Reckoning)", fontweight='bold')
    ax2.set_xlabel("X Koordinatı (Metre)")
    ax2.set_ylabel("Y Koordinatı (Metre)")
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':')

    # 3. LiDAR Grafiği (Satır 1, Sütun 0) - Polar koordinat tarzı ama Kartezyen eksen
    ax3 = plt.subplot(2, 3, 4)
    acilar_derece = np.degrees(lidar_acilar)
    ax3.plot(acilar_derece, lidar_ham, color='gray', alpha=0.6, linestyle='-', label='Ham LiDAR Verisi')
    
    # Filtrelenmiş / Kümelenmiş noktalar
    renkler = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    for i, kume in enumerate(lidar_kume):
        kx = [np.degrees(n[2]) for n in kume]
        ky = [n[1] for n in kume]
        ax3.scatter(kx, ky, color=renkler[i % len(renkler)], s=30, label=f'Tespit: Engel {i+1}', zorder=5)
        
    ax3.axhline(y=6.0, color='r', linestyle='--', alpha=0.5, label='Eşik (Threshold)')
    ax3.set_title("3. LiDAR Grafiği\n(Ham Sensör vs Filtrelenmiş Kümeler)", fontweight='bold')
    ax3.set_xlabel("Işın Açısı (Derece)")
    ax3.set_ylabel("Ölçülen Mesafe (Metre)")
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, linestyle=':')

    # 4. Zaman Serisi Lokalizasyon Grafikleri (Satır 1, Sütun 1)
    ax4 = plt.subplot(2, 3, 5)
    ax4.plot(zaman_dizisi, kayit_gercek['x'], 'g-', alpha=0.5, label='x_gerçek(t)')
    ax4.plot(zaman_dizisi, kayit_ekf['x'], 'g--', linewidth=2, label='x_tahmin(t)')
    
    ax4.plot(zaman_dizisi, kayit_gercek['y'], 'b-', alpha=0.5, label='y_gerçek(t)')
    ax4.plot(zaman_dizisi, kayit_ekf['y'], 'b--', linewidth=2, label='y_tahmin(t)')
    
    # Açıyı sağ Y ekseninde gösterelim
    ax4_right = ax4.twinx()
    derece_gercek = np.degrees(kayit_gercek['teta'])
    derece_ekf = np.degrees(kayit_ekf['teta'])
    ax4_right.plot(zaman_dizisi, derece_gercek, 'r-', alpha=0.3, label='theta_gerçek(t)')
    ax4_right.plot(zaman_dizisi, derece_ekf, 'r:', linewidth=2, label='theta_tahmin(t)')
    ax4_right.set_ylabel("Yönelim Açısı (Derece)", color='r')
    ax4_right.tick_params(axis='y', labelcolor='r')
    
    ax4.set_title("4. Zaman Serisi Lokalizasyon Grafikleri\n(x(t), y(t) ve theta(t))", fontweight='bold')
    ax4.set_xlabel("Zaman (Saniye)")
    ax4.set_ylabel("Konum (Metre)")
    
    # İki eksenin de legend'ını birleştir
    lines_1, labels_1 = ax4.get_legend_handles_labels()
    lines_2, labels_2 = ax4_right.get_legend_handles_labels()
    ax4.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=8)
    ax4.grid(True, linestyle=':')

    # 5. Hata Grafiği (Satır 1, Sütun 2)
    ax5 = plt.subplot(2, 3, 6)
    ax5.plot(zaman_dizisi, hata_dizisi, color='purple', linewidth=2.5, label='Lokalizasyon Konum Hatası (EKF)')
    
    # Ortalama hatayı (MAE) çizgi olarak ekle
    ortalama_hata = np.mean(hata_dizisi)
    ax5.axhline(y=ortalama_hata, color='orange', linestyle='--', linewidth=2, label=f'Ortalama Hata (MAE: {ortalama_hata:.2f} m)')
    
    ax5.fill_between(zaman_dizisi, hata_dizisi, color='purple', alpha=0.2) # Alanı boya
    
    ax5.set_title("5. Hata Grafiği\n(Zaman Boyunca Konum Hatası)", fontweight='bold')
    ax5.set_xlabel("Zaman (Saniye)")
    ax5.set_ylabel("Mesafe Sapması / Hata (Metre)")
    ax5.legend(loc='upper right')
    ax5.grid(True, linestyle=':')

    # Grafikleri sıkıştır ve göster
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Ana başlık için üstte biraz boşluk bırak
    print("Grafik rapor panosu ekrana çizdiriliyor...")
    plt.show()

if __name__ == "__main__":
    grafik_raporlarini_olustur()
