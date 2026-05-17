import numpy as np
import matplotlib.pyplot as plt

class DiferansiyelSurusluRobot:
    """
    Non-holonomic kısıtlara sahip klasik Diferansiyel Sürüşlü (Differential Drive) mobil robot kinematik modeli.
    Bu modelde robot yana doğru doğrudan kayma hareketi (crab steering) yapamaz.
    """
    def __init__(self, baslangic_x=0.0, baslangic_y=0.0, baslangic_teta=0.0):
        """
        :param baslangic_x: Başlangıç X konumu (metre)
        :param baslangic_y: Başlangıç Y konumu (metre)
        :param baslangic_teta: Başlangıç yönelimi / açısı (radyan)
        """
        # Robotun Durum Vektörü [x, y, theta]
        self.x = baslangic_x
        self.y = baslangic_y
        self.teta = baslangic_teta
        
        # Fiziksel kısıtlar (Hız limitleri)
        self.maksimum_cizgisel_hiz = 2.0  # m/s
        self.maksimum_acisal_hiz = np.pi  # rad/s (~180 derece/saniye)
        
        # Çizim ve analiz için geçmiş durumları kaydetme
        self.gecmis_x = [self.x]
        self.gecmis_y = [self.y]
        self.gecmis_teta = [self.teta]

    def durum_guncelle(self, v, omega, dt=0.1):
        """
        Belirtilen hız komutlarına göre robotun kinematik denklemlerle yeni konumunu hesaplar.
        
        Hareket Denklemleri (Non-holonomic kısıtlar):
        x(t+1) = x(t) + v * cos(theta(t)) * dt
        y(t+1) = y(t) + v * sin(theta(t)) * dt
        theta(t+1) = theta(t) + omega * dt
        
        :param v: Çizgisel hız (m/s)
        :param omega: Açısal hız (rad/s)
        :param dt: Zaman adımı (saniye)
        :return: Yeni durum (x, y, teta)
        """
        # Hız komutlarını robotun fiziksel limitlerine sınırlandırma
        v = np.clip(v, -self.maksimum_cizgisel_hiz, self.maksimum_cizgisel_hiz)
        omega = np.clip(omega, -self.maksimum_acisal_hiz, self.maksimum_acisal_hiz)

        # Euler metodu ile kinematik modelin entegrasyonu
        yeni_x = self.x + v * np.cos(self.teta) * dt
        yeni_y = self.y + v * np.sin(self.teta) * dt
        yeni_teta = self.teta + omega * dt
        
        # Açıyı [-pi, pi] aralığına normalize etme
        yeni_teta = np.arctan2(np.sin(yeni_teta), np.cos(yeni_teta))
        
        # Durumu güncelle
        self.x = yeni_x
        self.y = yeni_y
        self.teta = yeni_teta
        
        # Geçmişe ekle
        self.gecmis_x.append(self.x)
        self.gecmis_y.append(self.y)
        self.gecmis_teta.append(self.teta)
        
        return self.x, self.y, self.teta

def kinematik_testi_calistir():
    """
    Oluşturulan Diferansiyel Sürüşlü robot modelinin doğru çalışıp çalışmadığını
    test etmek için örnek bir hareket senaryosu uygular ve görselleştirir.
    """
    # Robotu başlangıç noktasına yerleştir
    robot = DiferansiyelSurusluRobot(baslangic_x=2.0, baslangic_y=2.0, baslangic_teta=np.pi/4)
    
    dt = 0.1 # 100 ms zaman adımı
    
    # Senaryo komutları listesi: (süre, cizgisel_hiz(v), acisal_hiz(w))
    senaryo = [
        (3.0, 1.5, 0.0),    # 3 sn boyunca düz git
        (4.0, 1.5, -0.5),   # 4 sn boyunca sağa dönerek (negatif açısal hız) git
        (3.0, 1.0, 0.8),    # 3 sn boyunca sola dönerek (pozitif açısal hız) git
        (2.0, -1.0, 0.0)    # 2 sn boyunca geriye doğru düz git
    ]
    
    zaman = 0.0
    for sure, v, w in senaryo:
        adim_sayisi = int(sure / dt)
        for _ in range(adim_sayisi):
            robot.durum_guncelle(v, w, dt)
            zaman += dt

    # Sonuçları görselleştirme
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(robot.gecmis_x, robot.gecmis_y, 'b-', linewidth=2.5, label="Robot Güzergahı")
    ax.plot(robot.gecmis_x[0], robot.gecmis_y[0], 'go', markersize=10, label="Başlangıç Noktası")
    ax.plot(robot.gecmis_x[-1], robot.gecmis_y[-1], 'ro', markersize=10, label="Bitiş Noktası")
    
    # Robotun yönelimini (heading) belirli aralıklarla oklarla göster
    ok_araligi = 15
    for i in range(0, len(robot.gecmis_x), ok_araligi):
        x = robot.gecmis_x[i]
        y = robot.gecmis_y[i]
        teta = robot.gecmis_teta[i]
        
        # Yönelim oku
        ax.arrow(x, y, 0.5 * np.cos(teta), 0.5 * np.sin(teta), 
                 head_width=0.2, head_length=0.2, fc='red', ec='red', alpha=0.6)

    plt.title("Adım 2: Non-Holonomic Diferansiyel Sürüşlü Robot Kinematik Testi", pad=15)
    plt.xlabel("X Ekseni Konumu (m)")
    plt.ylabel("Y Ekseni Konumu (m)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.axis('equal') # X ve Y eksen oranlarını eşitle ki dönüşler düzgün görünsün
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("--- ADIM 2: KİNEMATİK MODEL TESTİ ---")
    print("Diferansiyel sürüşlü robot modeli oluşturuldu.")
    print("Örnek hareket senaryosu çalıştırılıyor (Düz, Sağa dönüş, Sola dönüş, Geri vites)...")
    kinematik_testi_calistir()
    print("Test tamamlandı. Çıktı grafiği gösteriliyor.")
