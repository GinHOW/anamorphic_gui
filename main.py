import sys
import os
import tkinter as tk

# 将当前目录加入系统路径以确保顺利导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gui import AnamorphicApp

def main():
    root = tk.Tk()
    app = AnamorphicApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
