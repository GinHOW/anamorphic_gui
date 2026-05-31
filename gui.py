import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import os
import sys

# 确保能导入同目录下的 anamorphic.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import anamorphic

class AnamorphicApp:
    def __init__(self, root):
        self.root = root
        self.root.title("球面镜面反射逆向映射生成器 (Spherical Anamorphism)")
        self.root.geometry("1280x880")
        self.root.minsize(1100, 750)
        
        # 数据变量
        self.src_full = None       # 高清原始 OpenCV 图像 (可能是 3通道 BGR 或 4通道 BGRA)
        self.src_preview = None    # 缩放后的实时预览源图 (同样维持通道数)
        self.distorted_full = None # 高清扭曲 OpenCV 结果图
        
        # 动态自适应预览尺寸的默认大小 (会被 resize 事件动态覆盖)
        self.max_preview_w = 400
        self.max_preview_h = 400
        
        self.input_path = ""
        
        # 配置美观 of ttk 样式
        self.setup_styles()
        
        # 创建主界面布局 (采用双栏可手动拉拽调节的 PanedWindow)
        self.create_layout()
        
        # 初始化默认数据 (使用动态文字模式生成默认文字)
        self.apply_text_input()
        
        # 绑定右边预览栏的尺寸改变事件，实现完美的自适应动态缩放
        self.preview_panel.bind("<Configure>", self.on_preview_panel_resize)

    def setup_styles(self):
        self.style = ttk.Style()
        # 使用 macOS 的 native 主题或者 clean 默认主题
        if sys.platform == "darwin":
            self.style.theme_use("aqua")
            
        # 自定义一些样式
        self.style.configure("Header.TLabel", font=("Helvetica", 16, "bold"), foreground="#2c3e50")
        self.style.configure("SubHeader.TLabel", font=("Helvetica", 12, "bold"), foreground="#34495e")
        self.style.configure("Accent.TButton", font=("Helvetica", 11, "bold"), foreground="#2980b9")
        self.style.configure("Big.TButton", font=("Helvetica", 12, "bold"))

    def create_layout(self):
        # 采用 tk.PanedWindow 替代 ttk.PanedWindow，以便精确设置 minsize (最小尺寸限制) 和 stretch (拉伸策略)
        # 这可以 100% 避免左边控制栏被自动压缩隐藏或消失！
        self.paned_container = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bd=0, sashwidth=6, bg="#cfd8dc", relief="flat")
        self.paned_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # --- 左侧控制面板 (包装在外部Frame内作为PanedWindow的一个槽) ---
        sidebar_outer = ttk.Frame(self.paned_container, padding="10")
        
        # add 属性说明：
        # - minsize=390: 左侧栏绝对不允许被压缩到 390 像素以下，彻底根治消失问题！
        # - stretch="never": 当拉大软件主窗口时，左侧控制面板大小不改变，仅拉伸右侧预览区。
        self.paned_container.add(sidebar_outer, minsize=390, stretch="never")
        
        sidebar = ttk.Frame(sidebar_outer)
        sidebar.pack(fill="both", expand=True)
        
        # 头部标题
        header = ttk.Label(sidebar, text="球面反射逆向映射", style="Header.TLabel")
        header.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        
        # 1. 原始输入区域 (支持图片上传 & 直接输入文字)
        input_lf = ttk.LabelFrame(sidebar, text=" 1. 原始画面输入 ", padding="10")
        input_lf.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        input_lf.columnconfigure(1, weight=1)
        
        # 图片上传行
        self.load_btn = ttk.Button(input_lf, text="选择本地图片", command=self.browse_file)
        self.load_btn.grid(row=0, column=0, sticky="w", padx=(0, 5), pady=2)
        
        # 文字输入行
        ttk.Label(input_lf, text="或直接输入文字:").grid(row=1, column=0, sticky="w", pady=5)
        self.text_var = tk.StringVar(value="ANAMORPHIC")
        self.text_entry = ttk.Entry(input_lf, textvariable=self.text_var, width=15)
        self.text_entry.grid(row=1, column=1, sticky="ew", padx=(0, 5), pady=2)
        self.text_entry.bind("<Return>", lambda e: self.apply_text_input())
        
        self.text_btn = ttk.Button(input_lf, text="渲染文字", command=self.apply_text_input)
        self.text_btn.grid(row=1, column=2, sticky="e", pady=2)
        
        # 当前状态标签
        self.path_lbl = ttk.Label(input_lf, text="当前状态：正在使用文字", foreground="gray", font=("Helvetica", 10))
        self.path_lbl.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        # 2. 光学参数调节区域
        param_lf = ttk.LabelFrame(sidebar, text=" 2. 镜面与相机物理参数 ", padding="10")
        param_lf.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        param_lf.columnconfigure(1, weight=1)
        
        # 相机高度 H
        ttk.Label(param_lf, text="相机高度 (H):").grid(row=0, column=0, sticky="w", pady=5)
        self.h_var = tk.DoubleVar(value=400.0)
        self.h_slider = ttk.Scale(param_lf, from_=100, to=1000, variable=self.h_var, command=self.on_h_slider_move)
        self.h_slider.grid(row=0, column=1, sticky="ew", padx=10)
        self.h_spin = ttk.Spinbox(param_lf, from_=100, to=1000, width=6, textvariable=self.h_var, command=self.on_param_changed)
        self.h_spin.grid(row=0, column=2, sticky="e")
        
        # 半球半径 R
        ttk.Label(param_lf, text="镜面半径 (R):").grid(row=1, column=0, sticky="w", pady=5)
        self.r_var = tk.DoubleVar(value=200.0)
        self.r_slider = ttk.Scale(param_lf, from_=50, to=500, variable=self.r_var, command=self.on_r_slider_move)
        self.r_slider.grid(row=1, column=1, sticky="ew", padx=10)
        self.r_spin = ttk.Spinbox(param_lf, from_=50, to=500, width=6, textvariable=self.r_var, command=self.on_param_changed)
        self.r_spin.grid(row=1, column=2, sticky="e")
        
        # 边界尺寸 L
        ttk.Label(param_lf, text="投影半径 (L):").grid(row=2, column=0, sticky="w", pady=5)
        self.l_var = tk.DoubleVar(value=800.0)
        self.l_slider = ttk.Scale(param_lf, from_=200, to=2000, variable=self.l_var, command=self.on_l_slider_move)
        self.l_slider.grid(row=2, column=1, sticky="ew", padx=10)
        self.l_spin = ttk.Spinbox(param_lf, from_=200, to=2000, width=6, textvariable=self.l_var, command=self.on_param_changed)
        self.l_spin.grid(row=2, column=2, sticky="e")
        
        # 3. 投影模式与辅助线
        mode_lf = ttk.LabelFrame(sidebar, text=" 3. 映射投影配置 ", padding="10")
        mode_lf.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        mode_lf.columnconfigure(0, weight=1)
        
        self.mode_var = tk.StringVar(value="wrap")
        self.r_wrap = ttk.Radiobutton(mode_lf, text="360度极坐标环绕 (Wrap)", value="wrap", variable=self.mode_var, command=self.update_preview)
        self.r_wrap.grid(row=0, column=0, sticky="w", pady=2)
        self.r_billboard = ttk.Radiobutton(mode_lf, text="前方单侧投影 (Billboard)", value="billboard", variable=self.mode_var, command=self.update_preview)
        self.r_billboard.grid(row=1, column=0, sticky="w", pady=2)
        self.r_disk = ttk.Radiobutton(mode_lf, text="中心圆盘徽章投影 (Disk) - 适合圆环字/圆形图", value="disk", variable=self.mode_var, command=self.update_preview)
        self.r_disk.grid(row=2, column=0, sticky="w", pady=2)
        
        # 主白圈辅助线
        self.draw_circle_var = tk.BooleanVar(value=True)
        self.chk_circle = ttk.Checkbutton(mode_lf, text="在图像中心绘制半球底座边界参考线", variable=self.draw_circle_var, command=self.update_preview)
        self.chk_circle.grid(row=3, column=0, sticky="w", pady=(8, 2))

        # 新增：自定义多重同心参考圆控制横排
        ref_frame = ttk.Frame(mode_lf)
        ref_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(5, 2))
        
        ttk.Label(ref_frame, text="自定同心参考圆半径 (逗号隔开):").pack(side="left", padx=(0, 5))
        self.ref_circles_var = tk.StringVar(value="300, 500")
        self.ref_circles_entry = ttk.Entry(ref_frame, textvariable=self.ref_circles_var, width=12)
        self.ref_circles_entry.pack(side="left")
        # 绑定回车事件，自动渲染新参考线
        self.ref_circles_entry.bind("<Return>", lambda e: self.on_param_changed())

        # 4. 反射画面位置与大小微调
        pos_lf = ttk.LabelFrame(sidebar, text=" 4. 反射画面位置与大小微调 ", padding="10")
        pos_lf.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        pos_lf.columnconfigure(1, weight=1)

        # 缩放比例 Scale
        ttk.Label(pos_lf, text="画面缩放:").grid(row=0, column=0, sticky="w", pady=5)
        self.scale_var = tk.DoubleVar(value=1.0)
        self.scale_slider = ttk.Scale(pos_lf, from_=0.1, to=2.0, variable=self.scale_var, command=self.on_pos_slider_move)
        self.scale_slider.grid(row=0, column=1, sticky="ew", padx=10)
        self.scale_spin = ttk.Spinbox(pos_lf, from_=0.1, to=2.0, increment=0.1, width=6, textvariable=self.scale_var, command=self.on_param_changed)
        self.scale_spin.grid(row=0, column=2, sticky="e")

        # 水平偏移 Offset X
        ttk.Label(pos_lf, text="水平偏移 (X):").grid(row=1, column=0, sticky="w", pady=5)
        self.offx_var = tk.DoubleVar(value=0.0)
        self.offx_slider = ttk.Scale(pos_lf, from_=-1.0, to=1.0, variable=self.offx_var, command=self.on_pos_slider_move)
        self.offx_slider.grid(row=1, column=1, sticky="ew", padx=10)
        self.offx_spin = ttk.Spinbox(pos_lf, from_=-1.0, to=1.0, increment=0.1, width=6, textvariable=self.offx_var, command=self.on_param_changed)
        self.offx_spin.grid(row=1, column=2, sticky="e")

        # 垂直偏移 Offset Y
        ttk.Label(pos_lf, text="垂直偏移 (Y):").grid(row=2, column=0, sticky="w", pady=5)
        self.offy_var = tk.DoubleVar(value=0.0)
        self.offy_slider = ttk.Scale(pos_lf, from_=-1.0, to=1.0, variable=self.offy_var, command=self.on_pos_slider_move)
        self.offy_slider.grid(row=2, column=1, sticky="ew", padx=10)
        self.offy_spin = ttk.Spinbox(pos_lf, from_=-1.0, to=1.0, increment=0.1, width=6, textvariable=self.offy_var, command=self.on_param_changed)
        self.offy_spin.grid(row=2, column=2, sticky="e")

        # 绑定 Spinbox 回车事件
        for widget in (self.h_spin, self.r_spin, self.l_spin, self.scale_spin, self.offx_spin, self.offy_spin):
            widget.bind("<Return>", lambda e: self.on_param_changed())

        # 5. 执行控制与保存
        action_lf = ttk.LabelFrame(sidebar, text=" 5. 操作与导出 ", padding="10")
        action_lf.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        action_lf.columnconfigure(0, weight=1)
        
        self.live_preview_var = tk.BooleanVar(value=True)
        self.chk_live = ttk.Checkbutton(action_lf, text="启用实时拖动预览 (推荐)", variable=self.live_preview_var)
        self.chk_live.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        
        self.btn_save = ttk.Button(action_lf, text="导出高清印刷图片...", style="Big.TButton", command=self.export_high_res)
        self.btn_save.grid(row=1, column=0, columnspan=2, sticky="ew", ipady=8)
        
        # 物理限制提示
        hint_text = (
            "提示与交互说明：\n"
            "1. 【同心圆辅助线】：你可以输入多个物理半径，用逗号隔开并敲击回车，纸面上会生成高精度同心辅助圆，极其适合物理对准。\n"
            "2. 【手动拉伸中线】：你可以用鼠标拖动中线调节左栏和右侧预览窗的比例。"
        )
        self.hint_lbl = ttk.Label(sidebar, text=hint_text, justify="left", font=("Helvetica", 9), foreground="#7f8c8d")
        self.hint_lbl.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(5, 0))

        # --- 右侧预览面板 (加入到 PanedWindow 的右槽，设置 stretch="always" 使其成为主要的缩放区域) ---
        self.preview_panel = ttk.Frame(self.paned_container, padding="10")
        self.paned_container.add(self.preview_panel, stretch="always")
        
        self.preview_panel.columnconfigure(0, weight=1)
        self.preview_panel.columnconfigure(1, weight=1)
        self.preview_panel.rowconfigure(1, weight=1)
        
        # 左右预览模块标题
        self.lbl_src_title = ttk.Label(self.preview_panel, text="原始画面预览", style="SubHeader.TLabel")
        self.lbl_src_title.grid(row=0, column=0, pady=(0, 5))
        
        self.lbl_dst_title = ttk.Label(self.preview_panel, text="地面扭曲图预览 (反射后即变正常)", style="SubHeader.TLabel")
        self.lbl_dst_title.grid(row=0, column=1, pady=(0, 5))
        
        # 预览 Canvas / Label 容器
        self.canvas_src = tk.Label(self.preview_panel, bg="#f8f9fa", borderwidth=1, relief="solid")
        self.canvas_src.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        self.canvas_dst = tk.Label(self.preview_panel, bg="#f8f9fa", borderwidth=1, relief="solid")
        self.canvas_dst.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

    # --- 辅助参考线解析逻辑 ---
    def get_ref_radii(self):
        """解析文本框中逗号/空格分隔的自定义同心圆半径"""
        ref_str = self.ref_circles_var.get().strip()
        if not ref_str:
            return []
        try:
            cleaned = ref_str.replace(",", " ").replace(";", " ")
            return [float(x) for x in cleaned.split() if float(x) > 0]
        except ValueError:
            return []

    # --- 尺寸自适应逻辑 ---
    def on_preview_panel_resize(self, event):
        """当用户拖动中线、缩放窗口时，动态调整最大预览图宽度和高度"""
        w = event.width
        h = event.height
        
        target_w = (w - 40) // 2
        target_h = h - 60
        
        self.max_preview_w = max(100, target_w)
        self.max_preview_h = max(100, target_h)
        
        self.display_src_preview()
        self.update_preview()

    # --- 逻辑事件处理 ---
    def render_text_as_image(self, text, width=1600, height=400):
        """
        利用 PIL 动态渲染包含中英文的高质感文字图片，支持自动缩放字体以适应画布
        """
        img = Image.new("RGB", (width, height), (220, 220, 220))
        draw = ImageDraw.Draw(img)
        
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "Arial.ttf"
        ]
        font_path = None
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                break
                
        font_size = 180
        font = None
        
        while font_size > 20:
            if font_path:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                except:
                    font = ImageFont.load_default()
            else:
                font = ImageFont.load_default()
                break
                
            try:
                w, h = draw.textsize(text, font=font)
            except AttributeError:
                bbox = draw.textbbox((0, 0), text, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                
            if w < width - 100 and h < height - 60:
                break
            font_size -= 10
            
        try:
            w, h = draw.textsize(text, font=font)
        except AttributeError:
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
        text_x = (width - w) // 2
        text_y = (height - h) // 2
        
        draw.text((text_x + 4, text_y + 4), text, fill=(100, 100, 100), font=font)
        draw.text((text_x, text_y), text, fill=(255, 0, 0), font=font)
        
        opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return opencv_img

    def apply_text_input(self):
        """获取输入的文字并渲染至画布中"""
        text = self.text_var.get().strip()
        if not text:
            text = "ANAMORPHIC"
            self.text_var.set(text)
            
        self.src_full = self.render_text_as_image(text)
        
        preview_h = 200
        preview_w = 800
        self.src_preview = cv2.resize(self.src_full, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
        
        self.input_path = ""
        self.path_lbl.config(text=f"当前状态：使用渲染文字 「{text}」")
        
        self.display_src_preview()
        self.update_preview()

    def browse_file(self):
        """选择本地图片 (自动使用 IMREAD_UNCHANGED 保留 Alpha 透明通道)"""
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file_path:
            try:
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise Exception("读取图像文件失败，可能文件已损坏或格式不支持")
                
                self.src_full = img
                self.input_path = file_path
                self.path_lbl.config(text=f"已加载图片: {os.path.basename(file_path)}")
                
                # 制作适合实时映射的高效预览源图
                h, w = self.src_full.shape[:2]
                max_dim = 400
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    self.src_preview = cv2.resize(self.src_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    self.src_preview = self.src_full.copy()
                    
                self.scale_var.set(0.8)
                self.offx_var.set(0.0)
                self.offy_var.set(0.0)
                
                self.display_src_preview()
                self.update_preview()
            except Exception as e:
                messagebox.showerror("加载错误", f"无法加载该图片:\n{str(e)}")

    def display_src_preview(self):
        """在界面左侧显示原始图像的缩放预览 (处理透明通道)"""
        if self.src_full is None:
            return
            
        has_alpha = (len(self.src_full.shape) == 3 and self.src_full.shape[2] == 4)
        if has_alpha:
            img_rgb = cv2.cvtColor(self.src_full, cv2.COLOR_BGRA2RGBA)
            pil_img = Image.fromarray(img_rgb, "RGBA")
        else:
            img_rgb = cv2.cvtColor(self.src_full, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb, "RGB")
        
        pil_img.thumbnail((self.max_preview_w, self.max_preview_h))
        photo = ImageTk.PhotoImage(pil_img)
        
        self.canvas_src.config(image=photo)
        self.canvas_src.image = photo

    def on_h_slider_move(self, val):
        h = float(val)
        r = self.r_var.get()
        if h <= r:
            self.r_var.set(h - 1)
            self.r_spin.set(h - 1)
        self.on_param_changed()

    def on_r_slider_move(self, val):
        r = float(val)
        h = self.h_var.get()
        if r >= h:
            self.h_var.set(r + 1)
            self.h_slider.set(r + 1)
        self.on_param_changed()

    def on_l_slider_move(self, val):
        self.on_param_changed()

    def on_pos_slider_move(self, val):
        self.on_param_changed()

    def on_param_changed(self):
        """当数值文本框或滑块改变时触发，如果是实时预览就重新渲染"""
        if self.live_preview_var.get():
            self.update_preview()

    def update_preview(self):
        """重新计算并绘制实时的低分辨率扭曲图预览"""
        if self.src_preview is None:
            return
            
        h = self.h_var.get()
        r = self.r_var.get()
        l = self.l_var.get()
        mode = self.mode_var.get()
        draw_circle = self.draw_circle_var.get()
        
        scale = self.scale_var.get()
        offset_x = self.offx_var.get()
        offset_y = self.offy_var.get()
        
        # 解析自定义辅助参考线半径并等比缩放至预览图空间
        ref_radii = self.get_ref_radii()
        scale_factor = 250 / l
        preview_ref_radii = [r_val * scale_factor for r_val in ref_radii]
        
        preview_l = 250
        
        try:
            preview_result = anamorphic.generate_anamorphic(
                self.src_preview, 
                mode=mode, 
                H=h * (preview_l / l), 
                R=r * (preview_l / l), 
                L=preview_l,
                draw_circle=draw_circle,
                scale=scale,
                offset_x=offset_x,
                offset_y=offset_y,
                ref_radii=preview_ref_radii
            )
            
            # 渲染至右侧预览 Label
            has_alpha = (len(preview_result.shape) == 3 and preview_result.shape[2] == 4)
            if has_alpha:
                img_rgb = cv2.cvtColor(preview_result, cv2.COLOR_BGRA2RGBA)
                pil_img = Image.fromarray(img_rgb, "RGBA")
            else:
                img_rgb = cv2.cvtColor(preview_result, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb, "RGB")
                
            pil_img.thumbnail((self.max_preview_w, self.max_preview_h))
            photo = ImageTk.PhotoImage(pil_img)
            
            self.canvas_dst.config(image=photo)
            self.canvas_dst.image = photo
        except Exception as e:
            print(f"预览刷新出错: {e}")

    def export_high_res(self):
        """基于用户当前的完整物理坐标和高清原图，生成并导出最高画质的图像"""
        if self.src_full is None:
            messagebox.showwarning("警告", "请先输入文字或选择一张图片！")
            return
            
        h = self.h_var.get()
        r = self.r_var.get()
        l = self.l_var.get()
        mode = self.mode_var.get()
        draw_circle = self.draw_circle_var.get()
        
        scale = self.scale_var.get()
        offset_x = self.offx_var.get()
        offset_y = self.offy_var.get()
        
        # 获取最原始物理单位下的自定义参考线半径
        ref_radii = self.get_ref_radii()
        
        has_alpha = (len(self.src_full.shape) == 3 and self.src_full.shape[2] == 4)
        
        # 获取用户选择的文件保存路径
        initial_name = f"anamorphic_{mode}.png" if has_alpha else f"anamorphic_{mode}.jpg"
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png" if has_alpha else ".jpg",
            filetypes=[("PNG 图像 (支持透明)", "*.png"), ("JPEG 图像", "*.jpg"), ("所有文件", "*.*")],
            initialfile=initial_name,
            title="选择高清图片保存位置"
        )
        
        if not save_path:
            return
            
        # 显示生成中提示框
        progress_win = tk.Toplevel(self.root)
        progress_win.title("计算中")
        progress_win.geometry("300x100")
        progress_win.resizable(False, False)
        progress_win.grab_set()
        
        # 居中显示提示
        progress_win.update_idletasks()
        w = progress_win.winfo_width()
        h_win = progress_win.winfo_height()
        x = (progress_win.winfo_screenwidth() // 2) - (w // 2)
        y = (progress_win.winfo_screenheight() // 2) - (h_win // 2)
        progress_win.geometry(f'+{x}+{y}')
        
        ttk.Label(progress_win, text="正在使用高精度物理几何公式\n进行像素逆向光线追踪运算...", justify="center", font=("Helvetica", 11)).pack(pady=15)
        self.root.update()
        
        try:
            # 运行完整尺寸高清映射
            self.distorted_full = anamorphic.generate_anamorphic(
                self.src_full,
                mode=mode,
                H=h,
                R=r,
                L=l,
                draw_circle=draw_circle,
                scale=scale,
                offset_x=offset_x,
                offset_y=offset_y,
                ref_radii=ref_radii
            )
            
            # 保存
            cv2.imwrite(save_path, self.distorted_full)
            progress_win.destroy()
            
            # 提示成功
            messagebox.showinfo("导出成功", f"扭曲变形图已成功保存至:\n{save_path}\n\n建议打印后，在中心圆盘放置半径为 {int(r)} 等比长度的半球镜面进行反射观察！")
        except Exception as e:
            progress_win.destroy()
            messagebox.showerror("生成失败", f"生成高清图像时发生致命错误:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnamorphicApp(root)
    root.mainloop()
