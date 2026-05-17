import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings

# Animasyon sırasında çıkabilecek gereksiz uyarıları kapatmak için
warnings.filterwarnings("ignore", category=UserWarning)

# Projenin yapıtaşları olan modüller
from adim1_cevre_tasarimi import SimulasyonOrtami
from adim3_sensor_simulasyonu import SensorSistemi
from adim4_lokalizasyon_ekf import GenisletilmisKalmanFiltresi

def animasyonlu_simulasyon():
    ortam = SimulasyonOrtami()
    sensorler = SensorSistemi(ortam)
    
    # Haritalama Belleği
    harita_izgarasi = set()
    harita_sayaci = {} 
    
    plt.ion() 
    fig_anim, ax_anim = plt.subplots(figsize=(15, 8))
    
    # ==============================================================================
    # AŞAMA 1: DRONE İLE HIZLI STATİK HARİTALAMA (KEDİ HENÜZ ODADA YOK)
    # ==============================================================================
    print("\n>>> AŞAMA 1 BAŞLADI: Drone evi ÇOK YÜKSEK HIZDA tarıyor, lütfen bekleyin...")
    
    drone_rotasi = [
        (0.5, 0.5), (0.5, 14.5), 
        (29.5, 14.5), (29.5, 0.5), 
        (0.5, 0.5),
        
        (3.0, 0.5), (3.0, 14.5),
        (6.0, 14.5), (6.0, 0.5),
        (9.0, 0.5), (9.0, 14.5),
        (12.0, 14.5), (12.0, 0.5),
        (15.0, 0.5), (15.0, 14.5),
        (18.0, 14.5), (18.0, 0.5),
        (21.0, 0.5), (21.0, 14.5),
        (24.0, 14.5), (24.0, 0.5),
        (27.0, 0.5), (27.0, 14.5),
        
        (15.0, 7.5), (2.0, 6.0)    
    ]
    drone_x, drone_y = drone_rotasi[0]
    
    dt = 0.1
    drone_hizi_ms = 8.0 # Hızlandırılmış Drone Keşfi
    adim_sayaci = 0
    
    for wp in drone_rotasi[1:]:
        while True:
            mesafe = np.hypot(wp[0]-drone_x, wp[1]-drone_y)
            adim_mesafesi = drone_hizi_ms * dt
            
            if mesafe <= adim_mesafesi:
                drone_x, drone_y = wp[0], wp[1]
                break
            else:
                aci = np.arctan2(wp[1]-drone_y, wp[0]-drone_x)
                drone_x += adim_mesafesi * np.cos(aci) 
                drone_y += adim_mesafesi * np.sin(aci)
            
            # KEDİ YOK: Sadece sabit eşyalar taranıyor
            lidar_ham, lidar_acilar = sensorler.lidar_taramasi_yap(drone_x, drone_y, 0)
            for j in range(len(lidar_ham)):
                d = lidar_ham[j]
                if d < sensorler.maksimum_lidar_menzili - 0.1:
                    cx = drone_x + d * np.cos(lidar_acilar[j])
                    cy = drone_y + d * np.sin(lidar_acilar[j])
                    hucre = (round(cx, 1), round(cy, 1))
                    
                    harita_sayaci[hucre] = harita_sayaci.get(hucre, 0) + 1
                    if harita_sayaci[hucre] > 1:
                        harita_izgarasi.add(hucre)
            
            adim_sayaci += 1
            # Animasyon çok hızlı aksın diye ekranı sadece 5 adımda bir yeniliyoruz
            if adim_sayaci % 5 == 0:
                ax_anim.clear()
                ax_anim.set_xlim(0, ortam.genislik); ax_anim.set_ylim(0, ortam.yukseklik)
                ax_anim.set_title("AŞAMA 1: Yüksek Hızda Statik SLAM Haritalaması", fontweight='bold', fontsize=14)
                ax_anim.grid(True, linestyle=':', alpha=0.6)
                ax_anim.add_patch(patches.Rectangle((0, 0), ortam.genislik, ortam.yukseklik, fill=False, edgecolor='black', linewidth=4))
                ortam.engelleri_ciz(ax_anim)
                
                if harita_izgarasi:
                    mx, my = zip(*harita_izgarasi)
                    ax_anim.scatter(mx, my, c='black', s=8, marker='s', alpha=0.8, label='Sabit Eşya Hatları')
                    
                ax_anim.plot(drone_x, drone_y, 'c^', markersize=15, label='Hızlı Keşif Dronu')
                # Drone'un lazerleri soluk mavi
                for j in range(sensorler.lidar_isin_sayisi):
                    hx = drone_x + lidar_ham[j] * np.cos(lidar_acilar[j])
                    hy = drone_y + lidar_ham[j] * np.sin(lidar_acilar[j])
                    ax_anim.plot([drone_x, hx], [drone_y, hy], color='cyan', alpha=0.05, linewidth=1)
                    
                ax_anim.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
                plt.pause(0.001)

    print(">>> AŞAMA 1 BİTTİ: Evin %100 kusursuz krokisi çıkartıldı!\n")
    
    # ==============================================================================
    # KEDİNİN ODAYA GİRİŞİ (DİNAMİK ENGEL EKLENTİSİ)
    # ==============================================================================
    print(">>> DİKKAT: Odaya hareketli bir kedi girdi! Robot süpürge yola çıkıyor.")
    ortam.engeller.append((14.5, 5.0, 1.5, 1.5))
    ortam.engel_gorselleri.append({'isim': 'Hareketli Kedi', 'sekil': 'cember', 'renk': '#FF4500'})
    kedi_hizi = 2.0  
    kedi_yonu = 1    
    
    # ==============================================================================
    # AŞAMA 2: MİNİMAL İKONLU ROBOT İLE OTONOM NAVİGASYON VE DİNAMİK ENGELDEN KAÇIŞ
    # ==============================================================================
    toplam_zaman = 60.0
    adim_sayisi = int(toplam_zaman / dt)
    
    baslangic_x, baslangic_y, baslangic_teta = ortam.baslangic_noktasi[0], ortam.baslangic_noktasi[1], np.pi/4
    g_x, g_y, g_teta = baslangic_x, baslangic_y, baslangic_teta
    dr_x, dr_y, dr_teta = baslangic_x, baslangic_y, baslangic_teta
    ekf = GenisletilmisKalmanFiltresi(baslangic_x, baslangic_y, baslangic_teta, dt)
    
    ara_hedefler = [(7.0, 6.0), (12.0, 7.0), (17.0, 7.0), (22.0, 11.0), ortam.hedef_nokta]
    aktif_hedef_idx = 0
    
    zaman_dizisi, hata_dizisi = [], []
    kayit_gercek = {'x': [], 'y': [], 'teta': []}
    kayit_ekf = {'x': [], 'y': [], 'teta': []}
    kayit_dr = {'x': [], 'y': [], 'teta': []}
    
    for i in range(adim_sayisi):
        zaman = i * dt
        hedef_x, hedef_y = ara_hedefler[aktif_hedef_idx]
        
        # Kediyi Hareket Ettir
        kcx, kcy, kw, kh = ortam.engeller[-1]
        kcy += kedi_hizi * kedi_yonu * dt
        if kcy > 12.0: kcy, kedi_yonu = 12.0, -1
        elif kcy < 2.0: kcy, kedi_yonu = 2.0, 1
        ortam.engeller[-1] = (kcx, kcy, kw, kh)
        
        lidar_ham, lidar_acilar = sensorler.lidar_taramasi_yap(g_x, g_y, g_teta)
        
        # APF (Yapay Potansiyel Alanlar)
        f_x = 1.5 * (hedef_x - g_x)
        f_y = 1.5 * (hedef_y - g_y)
        mesafe = np.hypot(hedef_x - g_x, hedef_y - g_y)
        
        if mesafe > 1.0:
            f_x, f_y = (f_x / mesafe) * 2.0, (f_y / mesafe) * 2.0
            
        for j in range(len(lidar_ham)):
            d = lidar_ham[j]
            if 0.1 < d < 3.0: 
                mutlak_aci = g_teta + lidar_acilar[j]
                kuvvet = 3.0 * (1.0 / d - 1.0 / 3.0) * (1.0 / (d**2))
                f_x += kuvvet * np.cos(mutlak_aci + np.pi)
                f_y += kuvvet * np.sin(mutlak_aci + np.pi)
                
        if g_x < 1.0: f_x += 3.0 * (1.0/max(g_x, 0.1))**2
        if g_x > ortam.genislik - 1.0: f_x -= 3.0 * (1.0/max(ortam.genislik - g_x, 0.1))**2
        if g_y < 1.0: f_y += 3.0 * (1.0/max(g_y, 0.1))**2
        if g_y > ortam.yukseklik - 1.0: f_y -= 3.0 * (1.0/max(ortam.yukseklik - g_y, 0.1))**2
        
        istenen_aci = np.arctan2(f_y, f_x)
        aci_farki = istenen_aci - g_teta
        aci_farki = np.arctan2(np.sin(aci_farki), np.cos(aci_farki))
        
        # Aracın stabil hızı (Güvenli Sürüş)
        if abs(aci_farki) > np.pi / 3:
            v_komut, w_komut = 0.2, np.sign(aci_farki) * np.pi
        else:
            v_komut, w_komut = 2.0 * (1.0 - abs(aci_farki)/np.pi), 3.0 * aci_farki
            
        hedef_kabul = 1.5 if aktif_hedef_idx < len(ara_hedefler) - 1 else 0.3
        if mesafe < hedef_kabul:
            if aktif_hedef_idx < len(ara_hedefler) - 1:
                aktif_hedef_idx += 1
            else:
                break 
                
        # Kinematik ve EKF
        g_x += v_komut * np.cos(g_teta) * dt
        g_y += v_komut * np.sin(g_teta) * dt
        g_teta += w_komut * dt
        g_teta = np.arctan2(np.sin(g_teta), np.cos(g_teta))
        
        v_olculen, w_olculen = sensorler.tekerlek_enkoderi_oku(v_komut, w_komut)
        dr_x += v_olculen * np.cos(dr_teta) * dt
        dr_y += v_olculen * np.sin(dr_teta) * dt
        dr_teta += w_olculen * dt
        dr_teta = np.arctan2(np.sin(dr_teta), np.cos(dr_teta))
        
        ekf.tahmin_et(v_olculen, w_olculen)
        if i % 10 == 0:
            durum_est = ekf.guncelle([g_x + np.random.normal(0, 0.5), g_y + np.random.normal(0, 0.5), g_teta + np.random.normal(0, np.radians(5))])
        else:
            durum_est = ekf.x_est
            
        e_x, e_y, e_teta = durum_est[0, 0], durum_est[1, 0], durum_est[2, 0]
        
        zaman_dizisi.append(zaman)
        kayit_gercek['x'].append(g_x); kayit_gercek['y'].append(g_y); kayit_gercek['teta'].append(g_teta)
        kayit_dr['x'].append(dr_x); kayit_dr['y'].append(dr_y); kayit_dr['teta'].append(dr_teta)
        kayit_ekf['x'].append(e_x); kayit_ekf['y'].append(e_y); kayit_ekf['teta'].append(e_teta)
        hata_dizisi.append(np.hypot(g_x - e_x, g_y - e_y))
        
        if i % 3 == 0:
            # Görüntünün donmaması ve kedinin hareketinin kusursuz görünmesi için clear() ile çiziyoruz
            ax_anim.clear()
            ax_anim.set_xlim(0, ortam.genislik); ax_anim.set_ylim(0, ortam.yukseklik)
            ax_anim.set_title(f"AŞAMA 2: Dinamik Engel Aşma Simülasyonu | Adım: {i}", fontweight='bold', fontsize=14)
            ax_anim.grid(True, linestyle=':', alpha=0.6)
            
            ax_anim.add_patch(patches.Rectangle((0, 0), ortam.genislik, ortam.yukseklik, fill=False, edgecolor='black', linewidth=4))
            
            # Kediyi de barındıran tüm ortamı anlık konumuyla çizdir (Kedi hareket edecektir!)
            ortam.engelleri_ciz(ax_anim)
            
            # Statik Krokiyi arka planda saydam olarak göster
            if harita_izgarasi:
                mx, my = zip(*harita_izgarasi)
                ax_anim.scatter(mx, my, c='black', s=8, alpha=0.3, marker='s', label='Statik SLAM Krokisi')
            
            ax_anim.plot(ortam.baslangic_noktasi[0], ortam.baslangic_noktasi[1], 'go', markersize=10, label='Şarj İstasyonu')
            ax_anim.plot(ortam.hedef_nokta[0], ortam.hedef_nokta[1], 'r*', markersize=15, label='Nihai Hedef')
            
            ax_anim.plot(kayit_gercek['x'], kayit_gercek['y'], 'b-', linewidth=3, alpha=0.6, label='Gerçek Rota')
            ax_anim.plot(kayit_ekf['x'], kayit_ekf['y'], 'y--', linewidth=2, label='EKF Tahmini')
            
            # YENİ MİNİMAL VE ŞIK ROBOT İKONU ÇİZİMİ (Koyu Gri Gövde, Turkuaz Işıklar)
            robot_body = patches.Circle((g_x, g_y), radius=0.35, facecolor='#2C3E50', edgecolor='#1ABC9C', linewidth=2, zorder=10, label='Robot Süpürge')
            ax_anim.add_patch(robot_body)
            # Robotun yönünü belli eden turkuaz vizör çizgisi
            ax_anim.plot([g_x, g_x + 0.35 * np.cos(g_teta)], [g_y, g_y + 0.35 * np.sin(g_teta)], color='#1ABC9C', linewidth=3, zorder=11)
            ax_anim.plot([g_x], [g_y], 'o', color='#1ABC9C', markersize=3, zorder=12)
            
            # Robotun Lazer Işınlarını AÇIK MAVİ (Cyan) Yapıyoruz!
            for j in range(sensorler.lidar_isin_sayisi):
                m_aci = g_teta + lidar_acilar[j]
                hx = g_x + lidar_ham[j] * np.cos(m_aci)
                hy = g_y + lidar_ham[j] * np.sin(m_aci)
                ax_anim.plot([g_x, hx], [g_y, hy], color='cyan', alpha=0.25, linewidth=1)
                
            ax_anim.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
            plt.pause(0.001)

    print(">>> HEDEFE ULAŞILDI. Animasyon bitiriliyor, raporlar yükleniyor...\n")
    plt.ioff()
    plt.close(fig_anim) 

    lidar_kume = sensorler.lidar_veri_isleme(lidar_ham, lidar_acilar, esik_mesafesi=6.0)
    
    # =======================================================================
    # ------------------ RAPORLAMA VE ANALİZ PENCERESİ ------------------
    # =======================================================================
    fig_rapor = plt.figure(figsize=(18, 12))
    fig_rapor.suptitle("İKİ AŞAMALI (DRONE DESTEKLİ) KUSURSUZ HARİTALAMA VE NAVİGASYON RAPORU", fontsize=16, fontweight='bold', y=0.98)
    
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_xlim(0, ortam.genislik); ax1.set_ylim(0, ortam.yukseklik)
    ortam.engelleri_ciz(ax1)
    ax1.plot(ortam.baslangic_noktasi[0], ortam.baslangic_noktasi[1], 'go', markersize=10)
    ax1.plot(ortam.hedef_nokta[0], ortam.hedef_nokta[1], 'r*', markersize=15)
    ax1.set_title("1. Gerçek Ev Haritası (Kedi Dahil)", fontweight='bold')
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)"); ax1.grid(True, linestyle=':')

    ax2 = plt.subplot(2, 3, 2)
    ax2.set_xlim(0, ortam.genislik); ax2.set_ylim(0, ortam.yukseklik)
    ortam.engelleri_ciz(ax2)
    ax2.plot(kayit_dr['x'], kayit_dr['y'], 'r--', linewidth=2, alpha=0.7, label='Dead Reckoning')
    ax2.plot(kayit_gercek['x'], kayit_gercek['y'], 'b-', linewidth=4, alpha=0.6, label='Gerçek Rota')
    ax2.plot(kayit_ekf['x'], kayit_ekf['y'], 'y--', linewidth=2, label='EKF Kestirimi')
    ax2.plot(ortam.hedef_nokta[0], ortam.hedef_nokta[1], 'r*', markersize=20)
    ax2.set_title("2. Dinamik Kaçış ve Navigasyon Rotası", fontweight='bold')
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)"); ax2.legend(loc='lower left', fontsize=8); ax2.grid(True, linestyle=':')

    # İstenen Başlık Değişikliği
    ax3 = plt.subplot(2, 3, 3)
    if harita_izgarasi:
        mx, my = zip(*harita_izgarasi)
        ax3.scatter(mx, my, c='black', s=8, alpha=0.9, marker='s', label='Sabit Algılanan Yüzeyler')
    ax3.set_xlim(0, ortam.genislik); ax3.set_ylim(0, ortam.yukseklik)
    ax3.set_aspect('equal')
    ax3.set_title("3. Drone Tarafından Çıkarılan Statik SLAM Krokisi", fontweight='bold')
    ax3.set_xlabel("X (m)"); ax3.set_ylabel("Y (m)"); ax3.grid(True, linestyle=':', alpha=0.5); ax3.legend(loc='upper left', fontsize=8)

    ax4 = plt.subplot(2, 3, 4)
    acilar_derece = np.degrees(lidar_acilar)
    ax4.plot(acilar_derece, lidar_ham, color='gray', alpha=0.6, linestyle='-', label='Ham LiDAR')
    renkler = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    for i, kume in enumerate(lidar_kume):
        kx = [np.degrees(n[2]) for n in kume]
        ky = [n[1] for n in kume]
        ax4.scatter(kx, ky, color=renkler[i % len(renkler)], s=30, label=f'Engel {i+1}', zorder=5)
    ax4.axhline(y=6.0, color='r', linestyle='--', alpha=0.5, label='Eşik')
    ax4.set_title("4. Anlık LiDAR Sensör Verisi", fontweight='bold')
    ax4.set_xlabel("Açı (Derece)"); ax4.set_ylabel("Mesafe (m)"); ax4.legend(loc='upper right', fontsize=8); ax4.grid(True, linestyle=':')

    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(zaman_dizisi, kayit_gercek['x'], 'g-', alpha=0.5, label='x_gerçek')
    ax5.plot(zaman_dizisi, kayit_ekf['x'], 'g--', linewidth=2, label='x_tahmin')
    ax5.plot(zaman_dizisi, kayit_gercek['y'], 'b-', alpha=0.5, label='y_gerçek')
    ax5.plot(zaman_dizisi, kayit_ekf['y'], 'b--', linewidth=2, label='y_tahmin')
    ax5_right = ax5.twinx()
    ax5_right.plot(zaman_dizisi, np.degrees(kayit_gercek['teta']), 'r-', alpha=0.3, label='theta_gerçek')
    ax5_right.plot(zaman_dizisi, np.degrees(kayit_ekf['teta']), 'r:', linewidth=2, label='theta_tahmin')
    ax5.set_title("5. Zaman Serisi Grafikleri", fontweight='bold')
    ax5.set_xlabel("Zaman (sn)"); ax5.set_ylabel("Konum (m)"); ax5_right.set_ylabel("Yönelim (Derece)", color='r')
    lines_1, labels_1 = ax5.get_legend_handles_labels()
    lines_2, labels_2 = ax5_right.get_legend_handles_labels()
    ax5.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=8); ax5.grid(True, linestyle=':')

    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(zaman_dizisi, hata_dizisi, color='purple', linewidth=2.5, label='Hata (EKF)')
    ortalama_hata = np.mean(hata_dizisi)
    ax6.axhline(y=ortalama_hata, color='orange', linestyle='--', linewidth=2, label=f'MAE: {ortalama_hata:.2f} m')
    ax6.set_title("6. Hata Grafiği", fontweight='bold')
    ax6.set_xlabel("Zaman (sn)"); ax6.set_ylabel("Sapma (m)"); ax6.legend(loc='upper right'); ax6.grid(True, linestyle=':')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show() 

if __name__ == "__main__":
    animasyonlu_simulasyon()