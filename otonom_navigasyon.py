import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- AYARLAR ---
DT = 0.1
MAX_ITER = 2000 
GOAL_THRESHOLD = 0.8

class RobotOdeviKusursuzFinal:
    def __init__(self):
        # 30x10 Dikdörtgen Harita
        self.map_w = 30.0
        self.map_h = 10.0
        
        self.start = np.array([2.0, 2.0, np.pi/6]) 
        self.goal = np.array([29.0, 8.5])
        
        self.obstacles = [
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
        
        # Ara Hedefler (Waypoint Navigation)
        self.waypoints = [
            np.array([6.0, 4.0]),   
            np.array([10.0, 4.0]),  
            np.array([13.0, 1.5]),  
            np.array([16.0, 4.0]),  
            np.array([18.5, 5.5]),  
            np.array([22.0, 2.0]),  
            np.array([26.0, 1.5]),  
            self.goal               
        ]
        self.current_wp = 0
        
        self.true_state = self.start.copy()
        self.path_true = []
        self.path_est = []
        
        # Kalman Filtresi
        self.kf_x = np.array([[2.0], [2.0], [np.pi/6]])
        self.P = np.eye(3) * 0.1
        self.Q = np.diag([0.01, 0.01, 0.005]) 
        self.R = np.diag([0.1, 0.1, 0.05])    
        
        # LiDAR Analizi için örnek konum (Yolun ortasında bir yer)
        self.lidar_snapshot_pos = np.array([13.0, 1.5]) 
        self.lidar_raw = []
        self.lidar_filtered = []
        self.lidar_angles = []

    def get_dist_to_rect(self, px, py, cx, cy, w, h):
        dx = max(abs(px - cx) - w/2, 0)
        dy = max(abs(py - cy) - h/2, 0)
        return np.sqrt(dx**2 + dy**2)

    def motion_model(self, x, v, omega):
        new_x = x[0] + v * np.cos(x[2]) * DT
        new_y = x[1] + v * np.sin(x[2]) * DT
        new_theta = x[2] + omega * DT
        return np.array([new_x, new_y, new_theta])

    def kalman_step(self, v, omega, z):
        theta = self.kf_x[2, 0]
        self.kf_x[0, 0] += v * np.cos(theta) * DT
        self.kf_x[1, 0] += v * np.sin(theta) * DT
        self.kf_x[2, 0] += omega * DT
        self.P = self.P + self.Q
        H = np.eye(3)
        y = z.reshape(3,1) - (H @ self.kf_x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.kf_x = self.kf_x + (K @ y)
        self.P = (np.eye(3) - (K @ H)) @ self.P

    def get_control(self):
        curr_pos = self.kf_x[:2, 0]
        
        target = self.waypoints[self.current_wp]
        if np.linalg.norm(target - curr_pos) < 1.5 and self.current_wp < len(self.waypoints) - 1:
            self.current_wp += 1
            target = self.waypoints[self.current_wp]
            
        att_dir = (target - curr_pos) / max(np.linalg.norm(target - curr_pos), 0.1)
        att_force = att_dir * 1.5 
        
        rep_force = np.array([0.0, 0.0])
        for (cx, cy, w, h) in self.obstacles:
            dist = self.get_dist_to_rect(curr_pos[0], curr_pos[1], cx, cy, w, h)
            if dist < 1.5: 
                diff = curr_pos - np.array([cx, cy])
                rep_dir = diff / max(np.linalg.norm(diff), 0.001)
                rep_force += rep_dir * (1.0 / max(dist, 0.1)**2)
                
        if curr_pos[0] < 1.0: rep_force[0] += 2.0 / max(curr_pos[0], 0.1)
        if self.map_w - curr_pos[0] < 1.0: rep_force[0] -= 2.0 / max(self.map_w - curr_pos[0], 0.1)
        if curr_pos[1] < 1.0: rep_force[1] += 2.0 / max(curr_pos[1], 0.1)
        if self.map_h - curr_pos[1] < 1.0: rep_force[1] -= 2.0 / max(self.map_h - curr_pos[1], 0.1)

        total_force = att_force + rep_force
        target_angle = np.arctan2(total_force[1], total_force[0])
        angle_diff = (target_angle - self.kf_x[2, 0] + np.pi) % (2 * np.pi) - np.pi
        
        if abs(angle_diff) > np.pi / 2:
            v = 0.5
            omega = np.sign(angle_diff) * 2.5
        else:
            v = 1.8
            omega = 3.5 * angle_diff
        return v, omega

    def generate_lidar_scan(self):
        # 360 Derece Sanal LiDAR Taraması (Rapor için)
        self.lidar_angles = np.linspace(0, 2*np.pi, 72, endpoint=False) # 5 derece çözünürlük
        max_range = 5.0
        pos = self.lidar_snapshot_pos
        
        true_distances = []
        for ang in self.lidar_angles:
            d = max_range
            # Işın atma (Raycasting) simülasyonu
            for r in np.arange(0, max_range, 0.1):
                px = pos[0] + r * np.cos(ang)
                py = pos[1] + r * np.sin(ang)
                hit = False
                for (cx, cy, w, h) in self.obstacles:
                    if abs(px - cx) < w/2 and abs(py - cy) < h/2:
                        d = r
                        hit = True
                        break
                if hit: break
            true_distances.append(d)
        
        # 1. Ham Veri (Sensör Gürültüsü Eklenmiş)
        noise = np.random.normal(0, 0.25, len(true_distances))
        self.lidar_raw = np.clip(np.array(true_distances) + noise, 0, max_range)
        
        # 2. Filtrelenmiş Veri (Hareketli Ortalama / Düşük Geçiren Filtre)
        kernel_size = 3
        kernel = np.ones(kernel_size) / kernel_size
        self.lidar_filtered = np.convolve(self.lidar_raw, kernel, mode='same')

    def run(self):
        # Önce LiDAR verisini analiz et
        self.generate_lidar_scan()
        
        for i in range(MAX_ITER):
            v, omega = self.get_control()
            self.true_state = self.motion_model(self.true_state, v, omega)
            
            self.true_state[0] = np.clip(self.true_state[0], 0.2, self.map_w - 0.2)
            self.true_state[1] = np.clip(self.true_state[1], 0.2, self.map_h - 0.2)
            
            z = self.true_state + np.random.randn(3) @ np.sqrt(self.R) 
            self.kalman_step(v, omega, z)
            
            self.path_true.append(self.true_state[:2].copy())
            self.path_est.append(self.kf_x[:2, 0].copy())
            
            if np.linalg.norm(self.true_state[:2] - self.goal) < GOAL_THRESHOLD:
                print(f"SİMÜLASYON TAMAMLANDI! Araç hedefe ulaştı.")
                break
        
        self.draw_all()

    def draw_all(self):
        p_true = np.array(self.path_true)
        p_est = np.array(self.path_est)
        errors = np.linalg.norm(p_true - p_est, axis=1)

        # 3 Alt Grafiklik Profesyonel Layout
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])
        
        ax1 = fig.add_subplot(gs[0, :])   # Üstte tam genişlikte Harita
        ax2 = fig.add_subplot(gs[1, 0])   # Altta solda RMSE Hata
        ax3 = fig.add_subplot(gs[1, 1])   # Altta sağda LiDAR Analizi

        # --- GRAFİK 1: HARİTA (EKSEN ETİKETLİ) ---
        ax1.add_patch(patches.Rectangle((0, 0), self.map_w, self.map_h, fill=False, edgecolor='black', linewidth=3))
        for (cx, cy, w, h) in self.obstacles:
            ax1.add_patch(patches.Rectangle((cx-w/2, cy-h/2), w, h, color='#B0B0B0')) 
        
        ax1.plot(p_true[:,0], p_true[:,1], color='#3B59FF', linewidth=3, label='Örnek güvenli güzergah')
        ax1.plot(p_est[:,0], p_est[:,1], 'r--', linewidth=1, alpha=0.5, label='KF Tahmin')
        ax1.scatter(*self.start[:2], c='#00A000', s=200, zorder=5, label='Başlangıç') 
        ax1.scatter(*self.goal, c='#C00000', s=200, zorder=5, label='Hedef') 
        
        # EKSEN BİRİMLERİ EKLENDİ
        ax1.set_xlim(-1, self.map_w+1); ax1.set_ylim(-1, self.map_h+1)
        ax1.set_xlabel("X Mesafesi (m)", fontweight='bold')
        ax1.set_ylabel("Y Mesafesi (m)", fontweight='bold')
        ax1.legend(loc='upper left'); ax1.set_title("Şekil 1: Örnek Ortam Şekli (Eksenli ve Birimli)", fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.6)

        # --- GRAFİK 2: HATA ANALİZİ ---
        ax2.plot(errors, color='purple', linewidth=2)
        ax2.set_title(f"Lokalizasyon Hatası (RMSE: {np.sqrt(np.mean(errors**2)):.4f})", fontweight='bold')
        ax2.set_xlabel("Zaman Adımı")
        ax2.set_ylabel("Hata Mesafesi (m)")
        ax2.grid(True)

        # --- GRAFİK 3: LiDAR SENSÖR GÖRSELLEŞTİRMESİ ---
        # Açıları dereceye çevir
        angles_deg = np.degrees(self.lidar_angles)
        ax3.plot(angles_deg, self.lidar_raw, color='gray', alpha=0.7, label='Ham Veri (Gürültülü)', linewidth=1.5)
        ax3.plot(angles_deg, self.lidar_filtered, color='#FF8C00', label='Filtrelenmiş Veri', linewidth=2.5)
        ax3.set_title("360° LiDAR Sensör Tarama Analizi", fontweight='bold')
        ax3.set_xlabel("Tarama Açısı (Derece)")
        ax3.set_ylabel("Ölçülen Mesafe (m)")
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    sim = RobotOdeviKusursuzFinal()
    sim.run()