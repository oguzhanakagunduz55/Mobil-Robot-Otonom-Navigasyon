import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class SimulasyonOrtami:
    def __init__(self):
        # 1. Harita Oluşturma: Ev kat planı (30x15 boyutlarında)
        self.genislik = 30.0
        self.yukseklik = 15.0
        
        # 2. Noktalar: Şarj İstasyonu ve Temizlenecek Hedef Bölge (Koordinatlar sabit tutuldu)
        self.baslangic_noktasi = (2.0, 6.0)
        self.hedef_nokta = (28.0, 13.0)
        
        # 3. Engeller (Fizik Motoru Sınırları): 
        # DİKKAT: Diğer 7 dosyanın (LiDAR ve APF) hata vermemesi için orijinal (cx, cy, w, h)
        # sınır koordinatları birebir aynı bırakılmıştır. Çarpışma kutusu (Bounding Box) olarak çalışır.
        self.engeller = [
            (5.0, 4.0, 2.0, 2.0),     # E1
            (8.0, 10.0, 3.0, 1.5),    # E2
            (15.0, 12.0, 2.0, 2.0),   # E3
            (18.0, 3.0, 2.5, 2.5),    # E4
            (22.0, 8.0, 2.0, 3.0),    # E5
            (25.0, 4.0, 1.5, 2.0),    # E6
            (10.0, 2.0, 2.0, 1.5),    # E7
            (20.0, 13.0, 3.0, 1.0),   # E8
            (27.0, 9.0, 1.0, 3.0),    # E9
            (14.0, 8.0, 2.0, 2.0)     # E10
        ]
        
        # GÖRSELLEŞTİRME ÖZELLİKLERİ: Farklı şekiller ve renkler burada tanımlanır.
        # Bu liste sadece çizim yaparken kullanılır, fizik hesaplamalarını (Adım 3-4-5) bozmaz.
        self.engel_gorselleri = [
            {'isim': 'Yuvarlak Puf', 'sekil': 'cember', 'renk': '#8B4513'},
            {'isim': '3\'lü Koltuk', 'sekil': 'dikdortgen', 'renk': '#A0522D'},
            {'isim': 'Yemek Masası', 'sekil': 'dikdortgen', 'renk': '#CD853F'},
            {'isim': 'Büyük Saksı', 'sekil': 'cember', 'renk': '#2E8B57'},
            {'isim': 'Berjer', 'sekil': 'dikdortgen', 'renk': '#708090'},
            {'isim': 'TV Ünitesi', 'sekil': 'dikdortgen', 'renk': '#4682B4'},
            {'isim': 'Zigon Sehpa', 'sekil': 'dikdortgen', 'renk': '#DEB887'},
            {'isim': 'Kitaplık', 'sekil': 'dikdortgen', 'renk': '#5F9EA0'},
            {'isim': 'Gardırop', 'sekil': 'dikdortgen', 'renk': '#D2B48C'},
            {'isim': 'Yuvarlak Masa', 'sekil': 'cember', 'renk': '#808000'}
        ]
        
        # 4. Görev Tanımı
        self.gorev = "Şarj istasyonundan kalkan robot süpürgenin, evdeki\neşyalara (engellere) çarpmadan hedef bölgeyi temizlemesi."

    def engelleri_ciz(self, ax):
        """
        Bu fonksiyon, diğer dosyalardaki (Adım 7, Adım 8) grafiklere eşyaları
        farklı şekillerde (yuvarlak/dikdörtgen) ve isimleriyle çizmek için eklendi.
        """
        for i, (cx, cy, w, h) in enumerate(self.engeller):
            ozellik = self.engel_gorselleri[i]
            
            # Eğer şekil çember ise, genişliğin yarısını (w/2) yarıçap olarak al
            if ozellik['sekil'] == 'cember':
                engel_patch = patches.Circle((cx, cy), w/2, color=ozellik['renk'], alpha=0.85)
            # Değilse standart dikdörtgen eşya çiz
            else:
                alt_sol_x = cx - w / 2
                alt_sol_y = cy - h / 2
                engel_patch = patches.Rectangle((alt_sol_x, alt_sol_y), w, h, color=ozellik['renk'], alpha=0.85)
            
            ax.add_patch(engel_patch)
            
            # Eşyaların üzerine isimlerini yazdır
            ax.text(cx, cy, ozellik['isim'], ha='center', va='center', fontsize=8, color='white', fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

    def ortami_gorsellestir(self):
        """Oluşturulan ortamı matplotlib kullanarak görselleştirir."""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Harita sınırlarını ayarla
        ax.set_xlim(0, self.genislik)
        ax.set_ylim(0, self.yukseklik)
        
        # Harita dış çerçevesini çiz
        harita_cercevesi = patches.Rectangle((0, 0), self.genislik, self.yukseklik, fill=False, edgecolor='black', linewidth=4)
        ax.add_patch(harita_cercevesi)
        
        # Engelleri yardımcı fonksiyon ile çizdir
        self.engelleri_ciz(ax)

        # Başlangıç ve Hedef Noktalarını çiz
        ax.plot(self.baslangic_noktasi[0], self.baslangic_noktasi[1], 'go', markersize=12, label='Şarj İstasyonu (Başlangıç)')
        ax.plot(self.hedef_nokta[0], self.hedef_nokta[1], 'r*', markersize=15, label='Hedef Kirli Bölge')
        
        # Başlık ve Eksen Etiketleri
        plt.title(f"Adım 1: Robot Süpürge Çevre ve Senaryo Tasarımı\nGörev: {self.gorev}", pad=15, fontweight='bold')
        plt.xlabel("X Koordinatı / Ev Uzunluğu (m)")
        plt.ylabel("Y Koordinatı / Ev Genişliği (m)")
        
        # Lejant ve Grid
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Grafiği göster
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    ortam = SimulasyonOrtami()
    print("--- ADIM 1: ÇEVRE VE SENARYO TASARIMI BİLGİLERİ ---")
    print(f"Harita Boyutu   : {ortam.genislik}m x {ortam.yukseklik}m")
    print(f"Başlangıç       : {ortam.baslangic_noktasi}")
    print(f"Hedef Nokta     : {ortam.hedef_nokta}")
    print(f"Engel Sayısı    : {len(ortam.engeller)}")
    print(f"Görev Tanımı    : {ortam.gorev.replace(chr(10), ' ')}")
    print("-----------------------------------------------------")
    print("Harita görselleştiriliyor...")
    
    ortam.ortami_gorsellestir()