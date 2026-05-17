import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Önceki adımlardan modülleri içe aktar
from adim1_cevre_tasarimi import SimulasyonOrtami
from adim2_kinematik_model import DiferansiyelSurusluRobot
from adim3_sensor_simulasyonu import SensorSistemi

class NavigasyonSistemi:
    """
    Adım 5: Navigasyon ve Engellerden Kaçınma.
    Global Planner (Ara Hedefler) ve Local Planner (Yapay Potansiyel Alanlar - APF) içerir.
    """
    def __init__(self):
        # 1. Ortam ve Robot Kurulumu
        self.ortam = SimulasyonOrtami()
        self.robot = DiferansiyelSurusluRobot(
            baslangic_x=self.ortam.baslangic_noktasi[0], 
            baslangic_y=self.ortam.baslangic_noktasi[1], 
            baslangic_teta=np.pi/4
        )
        self.sensorler = SensorSistemi(self.ortam)
        
        # 2. Global Planner: Yol Planlama (Kaba ara hedefler - Waypoints)
        # Robotun hedefe (28, 13) gitmesi için haritadaki engellerin arasından kaba bir rota (yeni başlangıca göre)
        self.ara_hedefler = [
            (7.0, 6.0),
            (12.0, 7.0),
            (17.0, 7.0),
            (22.0, 11.0),
            self.ortam.hedef_nokta
        ]
        self.aktif_hedef_idx = 0
        self.hedef_kabul_yaricapi = 1.5 # Robot bu kadar yaklaşırsa hedefi ulaşıldı say

    def yapay_potansiyel_alan_hesapla(self, lidar_mesafeler, lidar_acilar):
        """
        Local Planner: APF algoritması ile çekici (hedefe) ve itici (engelden) kuvvetleri hesaplar.
        LiDAR verilerini kullanarak dinamik reaktif kaçınma sağlar.
        """
        hedef_x, hedef_y = self.ara_hedefler[self.aktif_hedef_idx]
        
        # --- A. ÇEKİCİ KUVVET (Attractive Force) ---
        k_att = 1.5 # Çekim katsayısı
        f_att_x = k_att * (hedef_x - self.robot.x)
        f_att_y = k_att * (hedef_y - self.robot.y)
        
        # Kuvvetin büyüklüğünü sınırla (robotun çok hızlı fırlamasını engellemek için)
        mesafe = np.hypot(hedef_x - self.robot.x, hedef_y - self.robot.y)
        if mesafe > 1.0:
            f_att_x = (f_att_x / mesafe) * 2.0
            f_att_y = (f_att_y / mesafe) * 2.0

        # --- B. İTİCİ KUVVET (Repulsive Force) ---
        k_rep = 3.0 # İtme katsayısı
        guvenli_mesafe = 3.0 # Bu mesafeden yakın olan engeller itmeye başlar
        
        f_rep_x = 0.0
        f_rep_y = 0.0
        
        # LiDAR verisini işleyerek itici kuvvet hesapla
        for i in range(len(lidar_mesafeler)):
            d = lidar_mesafeler[i]
            if 0.1 < d < guvenli_mesafe: # Çok yakın (0.1) değerler gürültü veya hata olabilir
                mutlak_aci = self.robot.teta + lidar_acilar[i]
                
                # İtme büyüklüğü formülü: Uzaklık azaldıkça kuvvet karesel olarak artar
                kuvvet = k_rep * (1.0 / d - 1.0 / guvenli_mesafe) * (1.0 / (d**2))
                
                # İtme yönü (engelden robota doğru, yani ışının tersi yönünde)
                f_rep_x += kuvvet * np.cos(mutlak_aci + np.pi)
                f_rep_y += kuvvet * np.sin(mutlak_aci + np.pi)
        
        # Harita sınırları için ekstra itici kuvvet (Duvarlara çarpmasını engellemek için)
        if self.robot.x < 1.0: f_rep_x += k_rep * (1.0/self.robot.x)**2
        if self.robot.x > self.ortam.genislik - 1.0: f_rep_x -= k_rep * (1.0/(self.ortam.genislik - self.robot.x))**2
        if self.robot.y < 1.0: f_rep_y += k_rep * (1.0/self.robot.y)**2
        if self.robot.y > self.ortam.yukseklik - 1.0: f_rep_y -= k_rep * (1.0/(self.ortam.yukseklik - self.robot.y))**2

        # --- C. TOPLAM KUVVET ---
        toplam_f_x = f_att_x + f_rep_x
        toplam_f_y = f_att_y + f_rep_y
        
        return toplam_f_x, toplam_f_y

    def kuvveti_hiza_donustur(self, f_x, f_y):
        """
        Hesaplanan toplam kuvvet vektörünü, Non-holonomic robotun anlayacağı
        çizgisel (v) ve açısal (omega) hız komutlarına dönüştürür.
        """
        # Kuvvet vektörünün yönü robotun gitmek istediği yönü belirtir
        istenen_aci = np.arctan2(f_y, f_x)
        
        # Robotun şu anki açısı ile istenen açı arasındaki fark
        aci_farki = istenen_aci - self.robot.teta
        aci_farki = np.arctan2(np.sin(aci_farki), np.cos(aci_farki)) # [-pi, pi] aralığına çek
        
        # Eğer çok dönmesi gerekiyorsa (engelden kaçıyorsa), hızı düşür ve dönmeye odaklan
        if abs(aci_farki) > np.pi / 3: # 60 dereceden büyük fark varsa
            v = 0.2
            omega = np.sign(aci_farki) * self.robot.maksimum_acisal_hiz
        else:
            v = self.robot.maksimum_cizgisel_hiz * (1.0 - abs(aci_farki)/np.pi) # Düz gittikçe hızlan
            omega = 3.0 * aci_farki # Oransal (P) kontrolcü
            
        return v, omega

    def navigasyonu_baslat(self):
        dt = 0.1
        maksimum_adim = 1000 # Simülasyonun sonsuz döngüye girmemesi için
        
        print("Navigasyon başlatıldı. Robot otonom olarak engellerden kaçarak hedefe gidiyor...")
        
        for adim in range(maksimum_adim):
            # 1. Sensörleri Oku (LiDAR verisi topla)
            lidar_mesafeler, lidar_acilar = self.sensorler.lidar_taramasi_yap(self.robot.x, self.robot.y, self.robot.teta)
            
            # 2. Local Planner (APF)
            f_x, f_y = self.yapay_potansiyel_alan_hesapla(lidar_mesafeler, lidar_acilar)
            
            # 3. Kinematik Kontrol (Kuvvet -> Hız dönüşümü)
            v_komut, w_komut = self.kuvveti_hiza_donustur(f_x, f_y)
            
            # 4. Robotu Hareket Ettir
            self.robot.durum_guncelle(v_komut, w_komut, dt)
            
            # 5. Global Planner Kontrolü (Hedefe ulaşıldı mı?)
            hedef_x, hedef_y = self.ara_hedefler[self.aktif_hedef_idx]
            mesafe = np.hypot(hedef_x - self.robot.x, hedef_y - self.robot.y)
            
            if mesafe < self.hedef_kabul_yaricapi:
                if self.aktif_hedef_idx < len(self.ara_hedefler) - 1:
                    print(f"Ara hedef {self.aktif_hedef_idx + 1} ulaşıldı! Bir sonraki hedefe yöneliniyor.")
                    self.aktif_hedef_idx += 1
                else:
                    print(f"\nGÖREV BAŞARILI! Robot ana hedef noktasına (Teslimat Noktası) güvenle ulaştı.")
                    print(f"Toplam adım sayısı: {adim}, Geçen süre: {adim*dt:.1f} saniye")
                    break
                    
        self.sonuclari_gorsellestir()

    def sonuclari_gorsellestir(self):
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Harita Sınırları ve Engeller
        ax.set_xlim(0, self.ortam.genislik)
        ax.set_ylim(0, self.ortam.yukseklik)
        for i, (cx, cy, w, h) in enumerate(self.ortam.engeller):
            ax.add_patch(patches.Rectangle((cx-w/2, cy-h/2), w, h, color='gray', alpha=0.6))
            
        # Global Planner Rotası (Ara Hedefler)
        wx = [self.ortam.baslangic_noktasi[0]] + [h[0] for h in self.ara_hedefler]
        wy = [self.ortam.baslangic_noktasi[1]] + [h[1] for h in self.ara_hedefler]
        ax.plot(wx, wy, 'y--', linewidth=2, label='Global Planner (Kaba Rota)')
        ax.scatter([h[0] for h in self.ara_hedefler], [h[1] for h in self.ara_hedefler], c='orange', s=50, marker='x')

        # Robotun İzlediği Gerçek Yol (Local Planner - APF sonucu)
        ax.plot(self.robot.gecmis_x, self.robot.gecmis_y, 'b-', linewidth=3, label='Otonom Navigasyon Yolu (APF)')
        
        # Başlangıç ve Hedef
        ax.plot(self.ortam.baslangic_noktasi[0], self.ortam.baslangic_noktasi[1], 'go', markersize=10, label='Başlangıç')
        ax.plot(self.ortam.hedef_nokta[0], self.ortam.hedef_nokta[1], 'ro', markersize=12, label='Ana Hedef')
        
        # Robotun son konumunu ve yönelimini çiz
        ax.arrow(self.robot.x, self.robot.y, np.cos(self.robot.teta), np.sin(self.robot.teta), 
                 head_width=0.6, head_length=0.6, fc='cyan', ec='blue')

        plt.title("Adım 5: Navigasyon ve Engellerden Kaçınma (Yapay Potansiyel Alanlar - APF)", pad=15, fontsize=14, fontweight='bold')
        plt.xlabel("X Koordinatı (m)")
        plt.ylabel("Y Koordinatı (m)")
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    navigasyon = NavigasyonSistemi()
    navigasyon.navigasyonu_baslat()
