"""IPA OCR GUI - 图形用户界面"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from PIL import Image, ImageTk
import threading



class IPAOCRGui:
    """IPA OCR图形界面"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("IPA OCR 识别系统")
        self.root.geometry("800x600")

        self.current_image = None
        self.current_image_path = None
        self.engine = None
        self.model_var = tk.StringVar(value="pix2tex")

        self._setup_ui()

    def _setup_ui(self):
        """设置UI布局"""
        # 顶部标题
        title_label = tk.Label(
            self.root,
            text="IPA 国际音标 OCR 识别系统",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)

        # 模型选择
        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=5)

        tk.Label(model_frame, text="选择模型:").pack(side=tk.LEFT, padx=5)

        model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=["pix2tex", "easyocr"],
            state="readonly",
            width=15
        )
        model_combo.pack(side=tk.LEFT, padx=5)

        # 图像显示区域
        self.image_frame = tk.Frame(self.root, bg="#f0f0f0", width=600, height=400)
        self.image_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(
            self.image_frame,
            text='点击"选择图像"按钮加载图片',
            bg="#f0f0f0",
            fg="#666666"
        )
        self.image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # 按钮区域
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        tk.Button(
            button_frame,
            text="选择图像",
            command=self._select_image,
            width=15
        ).pack(side=tk.LEFT, padx=5)

        self.recognize_btn = tk.Button(
            button_frame,
            text="开始识别",
            command=self._recognize,
            width=15,
            state=tk.DISABLED
        )
        self.recognize_btn.pack(side=tk.LEFT, padx=5)

        # 结果显示区域
        result_frame = tk.Frame(self.root)
        result_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        tk.Label(result_frame, text="识别结果:").pack(anchor=tk.W)

        self.result_text = tk.Text(result_frame, height=6, width=50)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # 置信度显示
        self.confidence_label = tk.Label(
            self.root,
            text="",
            fg="#666666"
        )
        self.confidence_label.pack(pady=5)

        # 状态栏
        self.status_label = tk.Label(
            self.root,
            text="就绪",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def _select_image(self):
        """选择图像文件"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[
                ("图像文件", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            self.current_image_path = file_path
            self._display_image(file_path)
            self.recognize_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"已加载: {Path(file_path).name}")

    def _display_image(self, image_path: str):
        """在界面上显示图像"""
        try:
            image = Image.open(image_path)

            # 调整图像大小以适应显示区域
            display_width = 600
            display_height = 350

            # 保持宽高比
            ratio = min(display_width / image.width, display_height / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))

            image = image.resize(new_size, Image.Resampling.LANCZOS)

            self.current_image = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.current_image, text="")

        except Exception as e:
            messagebox.showerror("错误", f"无法加载图像: {e}")

    def _recognize(self):
        """开始识别"""
        if not self.current_image_path:
            return

        self.recognize_btn.config(state=tk.DISABLED)
        self.status_label.config(text="正在识别...")
        self.result_text.delete(1.0, tk.END)
        self.confidence_label.config(text="")

        # 在后台线程中运行识别
        thread = threading.Thread(target=self._recognize_thread)
        thread.start()

    def _recognize_thread(self):
        """后台识别线程"""
        try:
            # 延迟导入以加快启动速度
            from ipa_ocr.engine import IPAOCREngine

            self.engine = IPAOCREngine(model=self.model_var.get())

            result = self.engine.recognize(
                self.current_image_path,
                return_confidence=True
            )

            if isinstance(result, tuple):
                text, confidence = result
            else:
                text = result
                confidence = 0.0

            # 更新UI
            self.root.after(0, self._update_result, text, confidence)

        except Exception as e:
            self.root.after(0, self._show_error, str(e))

    def _update_result(self, text: str, confidence: float):
        """更新识别结果"""
        self.result_text.insert(1.0, text)
        self.confidence_label.config(text=f"置信度: {confidence:.2%}")
        self.status_label.config(text="识别完成")
        self.recognize_btn.config(state=tk.NORMAL)

    def _show_error(self, error: str):
        """显示错误信息"""
        messagebox.showerror("识别错误", error)
        self.status_label.config(text="识别失败")
        self.recognize_btn.config(state=tk.NORMAL)

    def run(self):
        """运行GUI"""
        self.root.mainloop()


def main():
    """GUI入口点"""
    app = IPAOCRGui()
    app.run()


if __name__ == "__main__":
    main()
