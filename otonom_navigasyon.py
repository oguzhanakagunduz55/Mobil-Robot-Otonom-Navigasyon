import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- SİMÜLASYON PARAMETRELERİ ---
ZAMAN_ADIMI = 0.1
MAKSIMUM_ITERASYON = 2000 
HEDEF_YAKINLASMA_ESIGI = 0.8

class OtonomNavigasyonSimulasyonu:
    def __init__(self):
        # Çalışma Alanı Sınırları (30x10 m)
        self.harita_genislik = 30.0
        self.harita_yukseklik = 10.0
        
        self.baslangic_durumu = np.array([2.0, 2.0, np.pi/6]) 
        self.hedef_nokta = np.array([29.0, 8.5])
        
        # Engel Koordinatları (Merkez X, Merkez Y, Genişlik, Yükseklik)
        self.engeller = [
            (4.5, 7.0, 1.5, 3.5),   
            (7.0, 2.0, 2.5, 1.5),   
            (10.0, 8.0, 3.0, 1.5),  
            (13.0, 4.5, 1.5, 3.5),  
            (17.0, 7.5, 3.0, 1.5),  
            (18.5, 2.5, 1.5, 2.0),  
            (22.0, 5.0, 1.5, 3.5),  
            (24.5, 8.5, 2.5, 1.5),  
            (26.0, 3.0, 1.5, 2.0),  
            (28.5, 6.0, 1.0, 1.5)   
        ]
        
        # Rota Planlama için Ara Hedef Noktaları
        self.ara_hedefler = [
            np.array([6.0, 4.0]),   
            np.array([10.0, 4.0]),  
            np.array([13.0, 1.5]),  
            np.array([16.0, 4.0]),  
            np.array([18.5, 5.5]),  
            np.array([22.0, 2.0]),  
            np.array([26.0, 1.5]),  
            self.hedef_nokta               
        ]
        self.gecerli_ara_hedef = 0
        
        self.gercek_durum = self.baslangic_durumu.copy()
        self.gercek_guzergah = []
        self.kestirilen_guzergah = []
        
        # Genişletilmiş Kalman Filtresi (Durum Kestirimi) Parametreleri
        self.durum_kestirimi = np.array([[2.0], [2.0], [np.pi/6]])
        self.hata_kovaryans = np.eye(3) * 0.1
        self.surec_gurultu_kovaryans = np.diag([0.01, 0.01, 0.005]) 
        self.olcum_gurultu_kovaryans = np.diag([0.1, 0.1, 0.05])    
        
        # LiDAR Sensör Analizi için Referans Konum (Örneklem Noktası)
        self.lidar_orneklem_konumu = np.array([13.0, 1.5]) 
        self.lidar_ham_veri = []
        self.lidar_filtrelenmis_veri = []
        self.lidar_acilari = []

    def dikdortgene_olan_mesafeyi_hesapla(self, px, py, cx, cy, genislik, yukseklik):
        dx = max(abs(px - cx) - genislik/2, 0)
        dy = max(abs(py - cy) - yukseklik/2, 0)
        return np.sqrt(dx**2 + dy**2)

    def kinematik_hareket_modeli(self, durum, cizgisel_hiz, acisal_hiz):
        yeni_x = durum[0] + cizgisel_hiz * np.cos(durum[2]) * ZAMAN_ADIMI
        yeni_y = durum[1] + cizgisel_hiz * np.sin(durum[2]) * ZAMAN_ADIMI
        yeni_teta = durum[2] + acisal_hiz * ZAMAN_ADIMI
        return np.array([yeni_x, yeni_y, yeni_teta])

    def kalman_filtresi_guncelle(self, cizgisel_hiz, acisal_hiz, olcum):
        teta = self.durum_kestirimi[2, 0]
        self.durum_kestirimi[0, 0] += cizgisel_hiz * np.cos(teta) * ZAMAN_ADIMI
        self.durum_kestirimi[1, 0] += cizgisel_hiz * np.sin(teta) * ZAMAN_ADIMI
        self.durum_kestirimi[2, 0] += acisal_hiz * ZAMAN_ADIMI
        
        self.hata_kovaryans = self.hata_kovaryans + self.surec_gurultu_kovaryans
        H_matrisi = np.eye(3)
        hata_vektoru = olcum.reshape(3,1) - (H_matrisi @ self.durum_kestirimi)
        S_matrisi = H_matrisi @ self.hata_kovaryans @ H_matrisi.T + self.olcum_gurultu_kovaryans
        kalman_kazanci = self.hata_kovaryans @ H_matrisi.T @ np.linalg.inv(S_matrisi)
        
        self.durum_kestirimi = self.durum_kestirimi + (kalman_kazanci @ hata_vektoru)
        self.hata_kovaryans = (np.eye(3) - (kalman_kazanci @ H_matrisi)) @ self.hata_kovaryans

    def kontrol_sinyali_uret(self):
        mevcut_konum = self.durum_kestirimi[:2, 0]
        
        hedef = self.ara_hedefler[self.gecerli_ara_hedef]
        if np.linalg.norm(hedef - mevcut_konum) < 1.5 and self.gecerli_ara_hedef < len(self.ara_hedefler) - 1:
            self.gecerli_ara_hedef += 1
            hedef = self.ara_hedefler[self.gecerli_ara_hedef]
            
        cekim_yonu = (hedef - mevcut_konum) / max(np.linalg.norm(hedef - mevcut_konum), 0.1)
        cekim_kuvveti = cekim_yonu * 1.5 
        
        itme_kuvveti = np.array([0.0, 0.0])
        for (cx, cy, genislik, yukseklik) in self.engeller:
            mesafe = self.dikdortgene_olan_mesafeyi_hesapla(mevcut_konum[0], mevcut_konum[1], cx, cy, genislik, yukseklik)
            if mesafe < 1.5: 
                fark = mevcut_konum - np.array([cx, cy])
                itme_yonu = fark / max(np.linalg.norm(fark), 0.001)
                itme_kuvveti += itme_yonu * (1.0 / max(mesafe, 0.1)**2)
                
        if mevcut_konum[0] < 1.0: itme_kuvveti[0] += 2.0 / max(mevcut_konum[0], 0.1)
        if self.harita_genislik - mevcut_konum[0] < 1.0: itme_kuvveti[0] -= 2.0 / max(self.harita_genislik - mevcut_konum[0], 0.1)
        if mevcut_konum[1] < 1.0: itme_kuvveti[1] += 2.0 / max(mevcut_konum[1], 0.1)
        if self.harita_yukseklik - mevcut_konum[1] < 1.0: itme_kuvveti[1] -= 2.0 / max(self.harita_yukseklik - mevcut_konum[1], 0.1)

        toplam_kuvvet = cekim_kuvveti + itme_kuvveti
        hedef_aci = np.arctan2(toplam_kuvvet[1], toplam_kuvvet[0])
        aci_farki = (hedef_aci - self.durum_kestirimi[2, 0] + np.pi) % (2 * np.pi) - np.pi
        
        if abs(aci_farki) > np.pi / 2:
            cizgisel_hiz = 0.5
            acisal_hiz = np.sign(aci_farki) * 2.5
        else:
            cizgisel_hiz = 1.8
            acisal_hiz = 3.5 * aci_farki
        return cizgisel_hiz, acisal_hiz

    def lidar_tarama_simulasyonu_olustur(self):
        # 360 Derece Sanal LiDAR Tarama Simülasyonu
        self.lidar_acilari = np.linspace(0, 2*np.pi, 72, endpoint=False) # 5 derece çözünürlük
        maksimum_menzil = 5.0
        konum = self.lidar_orneklem_konumu
        
        gercek_mesafeler = []
        for aci in self.lidar_acilari:
            olculen_mesafe = maksimum_menzil
            # Işın İzleme (Raycasting) Algoritması
            for r in np.arange(0, maksimum_menzil, 0.1):
                px = konum[0] + r * np.cos(aci)
                py = konum[1] + r * np.sin(aci)
                carpisma_var_mi = False
                for (cx, cy, genislik, yukseklik) in self.engeller:
                    if abs(px - cx) < genislik/2 and abs(py - cy) < yukseklik/2:
                        olculen_mesafe = r
                        carpisma_var_mi = True
                        break
                if carpisma_var_mi: break
            gercek_mesafeler.append(olculen_mesafe)
        
        # 1. Gürültülü Sensör Verisi (Gauss Gürültüsü İlaveli)
        gurultu = np.random.normal(0, 0.25, len(gercek_mesafeler))
        self.lidar_ham_veri = np.clip(np.array(gercek_mesafeler) + gurultu, 0, maksimum_menzil)
        
        # 2. Filtrelenmiş Sensör Verisi (Hareketli Ortalama Filtresi)
        cekirdek_boyutu = 3
        cekirdek = np.ones(cekirdek_boyutu) / cekirdek_boyutu
        self.lidar_filtrelenmis_veri = np.convolve(self.lidar_ham_veri, cekirdek, mode='same')

    def simulasyonu_baslat(self):
        # Önce LiDAR verisini analiz et
        self.lidar_tarama_simulasyonu_olustur()
        
        for iterasyon in range(MAKSIMUM_ITERASYON):
            cizgisel_hiz, acisal_hiz = self.kontrol_sinyali_uret()
            self.gercek_durum = self.kinematik_hareket_modeli(self.gercek_durum, cizgisel_hiz, acisal_hiz)
            
            self.gercek_durum[0] = np.clip(self.gercek_durum[0], 0.2, self.harita_genislik - 0.2)
            self.gercek_durum[1] = np.clip(self.gercek_durum[1], 0.2, self.harita_yukseklik - 0.2)
            
            # Ölçüm modeline Gauss gürültüsü eklenmesi
            sensor_olcum_verisi = self.gercek_durum + np.random.randn(3) @ np.sqrt(self.olcum_gurultu_kovaryans) 
            self.kalman_filtresi_guncelle(cizgisel_hiz, acisal_hiz, sensor_olcum_verisi)
            
            self.gercek_guzergah.append(self.gercek_durum[:2].copy())
            self.kestirilen_guzergah.append(self.durum_kestirimi[:2, 0].copy())
            
            if np.linalg.norm(self.gercek_durum[:2] - self.hedef_nokta) < HEDEF_YAKINLASMA_ESIGI:
                print(f"Simülasyon Başarıyla Tamamlandı: Mobil robot hedef koordinata ulaştı.")
                break
        
        self.sonuclari_gorsellestir()

    def sonuclari_gorsellestir(self):
        gercek_p = np.array(self.gercek_guzergah)
        kestirilen_p = np.array(self.kestirilen_guzergah)
        hata_vektoru = np.linalg.norm(gercek_p - kestirilen_p, axis=1)

        # Grafiksel Arayüz (Subplot) Düzenlemesi
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])
        
        ax1 = fig.add_subplot(gs[0, :])   # Üstte tam genişlikte Harita
        ax2 = fig.add_subplot(gs[1, 0])   # Altta solda Hata Analizi
        ax3 = fig.add_subplot(gs[1, 1])   # Altta sağda LiDAR Analizi

        # --- GRAFİK 1: SİMÜLASYON ORTAMI VE GÜZERGAH ---
        ax1.add_patch(patches.Rectangle((0, 0), self.harita_genislik, self.harita_yukseklik, fill=False, edgecolor='black', linewidth=3))
        for (cx, cy, genislik, yukseklik) in self.engeller:
            ax1.add_patch(patches.Rectangle((cx-genislik/2, cy-yukseklik/2), genislik, yukseklik, color='#B0B0B0')) 
        
        ax1.plot(gercek_p[:,0], gercek_p[:,1], color='#3B59FF', linewidth=3, label='Gerçekleşen Güzergah (Referans)')
        ax1.plot(kestirilen_p[:,0], kestirilen_p[:,1], 'r--', linewidth=1, alpha=0.5, label='Kalman Filtresi Kestirimi')
        ax1.scatter(*self.baslangic_durumu[:2], c='#00A000', s=200, zorder=5, label='Başlangıç Noktası') 
        ax1.scatter(*self.hedef_nokta, c='#C00000', s=200, zorder=5, label='Hedef Nokta') 
        
        # Eksen Etiketleri ve Limitler
        ax1.set_xlim(-1, self.harita_genislik+1); ax1.set_ylim(-1, self.harita_yukseklik+1)
        ax1.set_xlabel("X Ekseni Konumu (m)", fontweight='bold')
        ax1.set_ylabel("Y Ekseni Konumu (m)", fontweight='bold')
        ax1.legend(loc='upper left'); ax1.set_title("Şekil 1: Otonom Navigasyon Ortamı ve İzlenen Güzergah", fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.6)

        # --- GRAFİK 2: LOKALİZASYON HATA ANALİZİ ---
        ax2.plot(hata_vektoru, color='purple', linewidth=2)
        ax2.set_title(f"Lokalizasyon Hatası (Kare Ortalama Karekök Hata: {np.sqrt(np.mean(hata_vektoru**2)):.4f})", fontweight='bold')
        ax2.set_xlabel("Zaman Adımı (K)")
        ax2.set_ylabel("Konum Hatası (m)")
        ax2.grid(True)

        # --- GRAFİK 3: LiDAR SENSÖR VERİ ANALİZİ ---
        # Açıları dereceye çevir
        acilar_derece = np.degrees(self.lidar_acilari)
        ax3.plot(acilar_derece, self.lidar_ham_veri, color='gray', alpha=0.7, label='Ham Sensör Verisi (Gauss Gürültülü)', linewidth=1.5)
        ax3.plot(acilar_derece, self.lidar_filtrelenmis_veri, color='#FF8C00', label='Filtrelenmiş Sinyal', linewidth=2.5)
        ax3.set_title("360° LiDAR Sensör Tarama Analizi", fontweight='bold')
        ax3.set_xlabel("Tarama Açısı (Derece)")
        ax3.set_ylabel("Ölçülen Mesafe (m)")
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    simulasyon = OtonomNavigasyonSimulasyonu()
    simulasyon.simulasyonu_baslat()
