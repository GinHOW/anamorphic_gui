import numpy as np
import cv2

def create_test_image(width=1600, height=400):
    """
    生成一个用于测试的"正常"图像，包含网格和文字。
    """
    img = np.full((height, width, 3), 220, dtype=np.uint8)
    
    # 绘制彩色网格，帮助观察畸变
    for x in range(0, width, 50):
        cv2.line(img, (x, 0), (x, height), (180, 180, 180), 2)
    for y in range(0, height, 50):
        cv2.line(img, (0, y), (width, y), (180, 180, 180), 2)
        
    font = cv2.FONT_HERSHEY_DUPLEX
    text = "ANAMORPHIC"
    text_size = cv2.getTextSize(text, font, 5, 12)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    # 添加文字阴影和主体
    cv2.putText(img, text, (text_x+5, text_y+5), font, 5, (100, 100, 100), 12)
    cv2.putText(img, text, (text_x, text_y), font, 5, (0, 0, 255), 12)
    return img

def generate_anamorphic(source_img, mode='wrap', H=400.0, R=200.0, L=800.0, draw_circle=True,
                        scale=1.0, offset_x=0.0, offset_y=0.0, ref_radii=None):
    """
    执行光学逆向映射，生成地面上的扭曲图像。
    支持自定义缩放、位置偏移以及多重同心参考圆线。
    
    :param source_img: 需要正常显示的原始图像 (可以是 3通道 BGR 或 4通道 BGRA)
    :param mode: 'wrap' (极坐标环绕), 'billboard' (单侧广告牌), 'disk' (中心圆盘/徽章模式)
    :param H: 观察者相机的高度
    :param R: 半球镜面的半径
    :param L: 生成的地面区域的半边长 (最终图像大小为 2L x 2L)
    :param draw_circle: 是否在图像中心绘制一个白色参考圆圈
    :param scale: 图像在镜子中呈现的缩放比例 (默认: 1.0)
    :param offset_x: 镜子中反射图像的水平偏移量 (Wrap模式下为旋转量)
    :param offset_y: 镜子中反射图像的垂直偏移量 (靠近/远离球顶)
    :param ref_radii: 自定义同心圆参考线的物理半径列表 (例如 [500.0, 600.0])
    :return: 生成的扭曲图像
    """
    ground_size = int(2 * L)
    
    # 检测输入图像的通道数，以支持透明底 PNG
    has_alpha = (len(source_img.shape) == 3 and source_img.shape[2] == 4)
    
    if has_alpha:
        # 4通道，初始化为完全透明底 (0, 0, 0, 0)
        ground_img = np.zeros((ground_size, ground_size, 4), dtype=np.uint8)
        border_val = (0, 0, 0, 0)
        circle_color = (255, 255, 255, 255)
        ref_circle_color = (180, 180, 180, 180) # 灰色透明辅助线
    else:
        # 3通道，初始化为黑色底 (0, 0, 0)
        ground_img = np.zeros((ground_size, ground_size, 3), dtype=np.uint8)
        border_val = (0, 0, 0)
        circle_color = (255, 255, 255)
        ref_circle_color = (150, 150, 150) # 灰色辅助线
    
    # 生成地面的物理坐标网格 Xg, Yg
    x = np.linspace(-L, L, ground_size)
    y = np.linspace(L, -L, ground_size)
    Xg, Yg = np.meshgrid(x, y)
    
    Rg = np.sqrt(Xg**2 + Yg**2)
    Psi = np.arctan2(Yg, Xg)
    
    # 防止 H <= R 的无解情况
    if H <= R:
        H = R + 1e-3
        
    # 计算临界角度
    c_max = R / H
    phi_max = np.arccos(c_max) 
    
    c_star = (R + np.sqrt(R**2 + 8*H**2)) / (4*H)
    phi_star = np.arccos(c_star)
    
    # 地面能被反射的最小半径 Rg_min
    Rg_min = H * R * np.sin(phi_max) / (R * np.cos(phi_max) - H * np.cos(2*phi_max))
    
    # 构建插值查找表：从地面半径 Rg 反推球面入射角 phi
    phi_table = np.linspace(phi_star + 1e-6, phi_max, 10000)
    Rg_table = H * R * np.sin(phi_table) / (R * np.cos(phi_table) - H * np.cos(2*phi_table))
    
    Rg_table_rev = Rg_table[::-1]
    phi_table_rev = phi_table[::-1]
    
    Phi_map = np.interp(Rg, Rg_table_rev, phi_table_rev)
    rv_map = R * np.sin(Phi_map) / (H - R * np.cos(Phi_map))
    
    # 确定有效视区的 rv 范围
    rv_max = R * np.sin(phi_max) / (H - R * np.cos(phi_max))
    max_visible_Rg = min(L * np.sqrt(2), Rg_table[0]) 
    phi_edge = np.interp(max_visible_Rg, Rg_table_rev, phi_table_rev)
    rv_min = R * np.sin(phi_edge) / (H - R * np.cos(phi_edge))
    
    src_h, src_w = source_img.shape[:2]
    aspect_ratio = src_w / src_h
    
    if mode == 'wrap':
        span = (rv_max - rv_min) * scale
        center_rv = (rv_max + rv_min) / 2.0 + offset_y * (rv_max - rv_min)
        
        rv_min_target = center_rv - span / 2.0
        rv_max_target = center_rv + span / 2.0
        
        map_y = (rv_map - rv_min_target) / (rv_max_target - rv_min_target) * src_h
        
        Psi_shifted = Psi + offset_x * np.pi
        map_x = (0.5 - (Psi_shifted - np.pi/2) / (2 * np.pi)) * src_w
        map_x = np.mod(map_x, src_w)
        
    elif mode == 'billboard':
        U = rv_map * np.cos(Psi)
        V = rv_map * np.sin(Psi)
        
        h_target = (rv_max - rv_min) * scale
        w_target = h_target * aspect_ratio
        
        y_center = (rv_max + rv_min) / 2.0 + offset_y * (rv_max - rv_min)
        x_center = offset_x * (rv_max - rv_min)
        
        map_y = (y_center + h_target / 2.0 - V) / h_target * src_h
        map_x = (U - (x_center - w_target / 2.0)) / w_target * src_w
        
    elif mode == 'disk':
        U = rv_map * np.cos(Psi)
        V = rv_map * np.sin(Psi)
        
        h_target = 2.0 * rv_max * scale
        w_target = h_target * aspect_ratio
        
        y_center = offset_y * rv_max
        x_center = offset_x * rv_max
        
        map_y = (y_center + h_target / 2.0 - V) / h_target * src_h
        map_x = (U - (x_center - w_target / 2.0)) / w_target * src_w
        
    else:
        raise ValueError("未知的映射模式")
        
    # 进行重投影插值 (使用具有透明通道的 border_val 填充越界部分)
    distorted_img = cv2.remap(source_img, map_x.astype(np.float32), map_y.astype(np.float32), 
                              interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
    
    # 遮罩掉无法产生反射的中心区域
    mask = Rg >= Rg_min
    ground_img[mask] = distorted_img[mask]
    
    # 1. 绘制自定义多重同心参考圆 (绘制在最底层，防止覆盖白色主圆圈)
    if draw_circle and ref_radii:
        for r_val in ref_radii:
            # 只有当参考圆半径在合法图像范围内时才绘制
            if 0 < r_val < L:
                # 绘制细灰色辅助线 (线宽为 1)
                cv2.circle(ground_img, (ground_size//2, ground_size//2), int(r_val), ref_circle_color, 1)
    
    # 2. 绘制白色主参考底座圆圈圈
    if draw_circle:
        cv2.circle(ground_img, (ground_size//2, ground_size//2), int(R), circle_color, max(2, int(R/100)))
    
    return ground_img
