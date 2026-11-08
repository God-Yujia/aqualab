"""AquaLab's local desktop preview and export."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from infer import IMAGE_EXTENSIONS, ROOT, UnderwaterEnhancer, rgb_image, strength_image
from local_control import limit_local_enhancement


BACKGROUND = "#0f1924"
PANEL = "#172635"
PREVIEW = "#0a131e"
TEXT = "#eef5f9"
MUTED = "#a8bfcd"
ACCENT = "#83e0d1"
FONT = "Microsoft YaHei UI"


class ImagePane(tk.Frame):
    def __init__(self, parent: tk.Widget, title: str, placeholder: str) -> None:
        super().__init__(parent, bg=PANEL, padx=12, pady=11)
        tk.Label(self, text=title, bg=PANEL, fg=TEXT, font=(FONT, 11, "bold")).pack(anchor="w", pady=(0, 8))
        self.canvas = tk.Canvas(self, bg=PREVIEW, highlightthickness=0, width=360, height=250)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._draw)
        self.placeholder = placeholder
        self.image = None
        self.preview = None

    def show(self, image: Image.Image | None) -> None:
        self.image = image
        self._draw()

    def _draw(self, event: tk.Event | None = None) -> None:
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width <= 1 or height <= 1:
            return
        self.canvas.delete("all")
        if self.image is None:
            self.preview = None
            self.canvas.create_text(width // 2, height // 2, text=self.placeholder, fill=MUTED, font=(FONT, 10))
        else:
            thumbnail = ImageOps.contain(self.image, (width, height), Image.Resampling.LANCZOS)
            self.preview = ImageTk.PhotoImage(thumbnail, master=self.canvas)
            self.canvas.create_image(width // 2, height // 2, image=self.preview)


def render_local(source, prediction, strength: float) -> tuple[Image.Image, Image.Image]:
    adjusted, coefficient = limit_local_enhancement(source, prediction, strength)
    return rgb_image(adjusted), strength_image(coefficient)


def interface_default() -> tuple[int, str]:
    try:
        manifest = json.loads((ROOT / "data/test_manifest.json").read_text(encoding="utf-8"))
        interface = manifest.get("protocol", {}).get("interface", {})
        value = interface.get("default_strength")
        count = interface.get("pair_count")
    except (OSError, ValueError, AttributeError):
        return 100, "默认 100%"
    if type(value) in (int, float) and 0 <= value <= 1:
        percent = round(100 * value)
        label = f"默认 {percent}%"
        if type(count) is int and count > 0:
            label += f" · 依据 {count} 对配对图实测"
        return percent, label
    return 100, "默认 100%"


class AquaLabWindow:
    def __init__(self, root: tk.Tk, checkpoint: Path, device: str) -> None:
        self.root = root
        self.checkpoint = checkpoint
        self.device = device
        self.enhancer = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.poll_job = None
        self.refresh_job = None
        self.closing = False
        self.tensors = None
        self.path = None
        self.result = None
        self.strength_map = None
        self.default_strength, default_label = interface_default()

        root.title("AquaLab · 水下图像增强")
        root.geometry(f"{min(1160, root.winfo_screenwidth() - 64)}x{min(735, root.winfo_screenheight() - 100)}")
        root.minsize(760, 480)
        root.configure(bg=BACKGROUND)
        root.protocol("WM_DELETE_WINDOW", self.close)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("Aqua.TButton", font=(FONT, 10, "bold"), padding=(16, 9), foreground=TEXT, background=PANEL)
        style.map("Aqua.TButton", foreground=[("disabled", "#748896")], background=[("active", "#264259")])
        style.configure("Open.TButton", font=(FONT, 10, "bold"), padding=(16, 9), foreground=BACKGROUND, background=ACCENT)
        style.map("Open.TButton", background=[("disabled", "#5e7e82"), ("active", "#a8f3e6")])
        style.configure("Map.TCheckbutton", background=PANEL, foreground=TEXT, font=(FONT, 10))
        style.map("Map.TCheckbutton", background=[("active", PANEL)], foreground=[("disabled", MUTED)])

        body = tk.Frame(root, bg=BACKGROUND, padx=24, pady=18)
        body.pack(fill="both", expand=True)
        header = tk.Frame(body, bg=BACKGROUND)
        header.pack(fill="x", pady=(0, 15))
        brand = tk.Frame(header, bg=BACKGROUND)
        brand.pack(side="left")
        tk.Label(brand, text="AquaLab", bg=BACKGROUND, fg=TEXT, font=("Segoe UI", 27, "bold")).pack(anchor="w")
        tk.Label(brand, text="本地水下图像增强", bg=BACKGROUND, fg=ACCENT, font=(FONT, 10)).pack(anchor="w")
        self.export_button = ttk.Button(header, text="导出 PNG", style="Aqua.TButton", command=self.save_result)
        self.export_button.pack(side="right")
        self.open_button = ttk.Button(header, text="打开照片", style="Open.TButton", command=self.choose_image)
        self.open_button.pack(side="right", padx=(0, 10))

        self.panels = tk.Frame(body, bg=BACKGROUND)
        self.panels.pack(fill="both", expand=True)
        self.panels.columnconfigure((0, 1), weight=1, uniform="preview")
        self.panels.rowconfigure(0, weight=1)
        self.original = ImagePane(self.panels, "原图", "打开一张水下照片")
        self.original.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.enhanced = ImagePane(self.panels, "增强结果", "处理完成后会显示结果")
        self.enhanced.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.map_pane = ImagePane(self.panels, "实际系数 · 黑 0% / 白 100%", "处理完成后会显示系数图")

        controls = tk.Frame(body, bg=PANEL, padx=14, pady=9)
        controls.pack(fill="x", pady=(12, 7))
        self.strength = tk.IntVar(root, self.default_strength)
        self.percent = tk.StringVar(root, f"{self.default_strength}%")
        self.show_map = tk.BooleanVar(root, False)
        self.map_toggle = ttk.Checkbutton(
            controls, text="显示局部控制", style="Map.TCheckbutton", variable=self.show_map,
            command=self._toggle_map,
        )
        self.map_toggle.pack(side="right", padx=(12, 0))
        self.map_export_button = ttk.Button(
            controls, text="导出系数图", style="Aqua.TButton",
            command=lambda: self.save_result(save_map=True),
        )
        tk.Label(controls, text="增强强度", bg=PANEL, fg=TEXT, font=(FONT, 10, "bold")).pack(side="left")
        tk.Label(controls, textvariable=self.percent, bg=PANEL, fg=ACCENT, width=6, font=(FONT, 10, "bold")).pack(side="right")
        self.scale = tk.Scale(
            controls, from_=0, to=100, orient="horizontal", showvalue=False, variable=self.strength,
            command=self.strength_changed, bg=PANEL, fg=TEXT, activebackground=ACCENT, troughcolor=PREVIEW,
            highlightthickness=0, relief="flat", sliderrelief="flat", sliderlength=22, width=13,
        )
        self.scale.pack(side="left", fill="x", expand=True, padx=12)
        self.status = tk.StringVar(root, "请选择照片。0% 显示处理尺寸下的原图，100% 为全强度局部控制。")
        footer = tk.Frame(body, bg=BACKGROUND)
        footer.pack(fill="x")
        tk.Label(footer, text=default_label, bg=BACKGROUND, fg=ACCENT, font=(FONT, 9)).pack(side="right", padx=(8, 0))
        tk.Label(footer, textvariable=self.status, anchor="w", bg=BACKGROUND, fg=MUTED, font=(FONT, 9)).pack(side="left", fill="x", expand=True)
        self._enable_result(False)

    def _enable_result(self, enabled: bool) -> None:
        self.scale.configure(state="normal" if enabled else "disabled")
        for control in (self.export_button, self.map_toggle, self.map_export_button):
            control.state(["!disabled" if enabled else "disabled"])

    def _toggle_map(self) -> None:
        if self.show_map.get():
            self.panels.columnconfigure(2, weight=1, uniform="preview")
            self.enhanced.grid_configure(padx=6)
            self.map_pane.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
            self.map_pane.show(self.strength_map)
            self.map_export_button.pack(side="right", before=self.map_toggle, padx=(12, 0))
        else:
            self.map_pane.grid_remove()
            self.panels.columnconfigure(2, weight=0, uniform="")
            self.enhanced.grid_configure(padx=(6, 0))
            self.map_export_button.pack_forget()

    def _reset_preview(self) -> None:
        self.path = self.tensors = self.result = self.strength_map = None
        self.strength.set(self.default_strength)
        self.percent.set(f"{self.default_strength}%")
        self.original.show(None)
        self.enhanced.show(None)
        self.map_pane.show(None)

    def choose_image(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root, title="选择水下照片",
            filetypes=[("图片", tuple(f"*{suffix}" for suffix in sorted(IMAGE_EXTENSIONS))), ("所有文件", "*.*")],
        )
        if selected:
            self.load_image(Path(selected))

    def load_image(self, path: Path) -> None:
        if self.closing or self.pending is not None:
            return
        self._finish_pending_preview()
        self.open_button.state(["disabled"])
        self._enable_result(False)
        self.status.set(f"正在处理 {path.name}…")
        self.pending = self.executor.submit(self._predict, path)
        self.poll_job = self.root.after(30, self._poll_prediction, path)

    def _predict(self, path: Path):
        with Image.open(path) as image:
            if self.enhancer is None:
                if not self.checkpoint.is_file():
                    raise FileNotFoundError("缺少 WWE-UIE 权重，请按 README 准备模型文件。")
                self.enhancer = UnderwaterEnhancer(self.checkpoint, self.device)
            original, source, prediction, _ = self.enhancer.predict(image)
        result, strength_map = render_local(source, prediction, self.default_strength / 100)
        return original, (source, prediction), result, strength_map

    def _poll_prediction(self, path: Path) -> None:
        self.poll_job = None
        if not self.pending.done():
            self.poll_job = self.root.after(30, self._poll_prediction, path)
            return
        if self.closing:
            self._finish_close()
            return
        future, self.pending = self.pending, None
        try:
            original, tensors, result, strength_map = future.result()
        except Exception as error:
            self._reset_preview()
            self.status.set(f"无法处理 {path.name}。")
            messagebox.showerror("处理失败", str(error), parent=self.root)
        else:
            self.path, self.tensors, self.result, self.strength_map = path, tensors, result, strength_map
            self.strength.set(self.default_strength)
            self.percent.set(f"{self.default_strength}%")
            self.original.show(original)
            self.enhanced.show(result)
            if self.show_map.get():
                self.map_pane.show(strength_map)
            self.status.set(f"{path.name}  ·  {result.width} × {result.height}  ·  可拖动滑杆调节")
        if not self.closing:
            self.open_button.state(["!disabled"])
            self._enable_result(self.result is not None)

    def strength_changed(self, _value: str) -> None:
        self.percent.set(f"{self.strength.get()}%")
        if not self.closing and self.tensors is not None and self.pending is None:
            if self.refresh_job is not None:
                self.root.after_cancel(self.refresh_job)
            self.refresh_job = self.root.after(35, self._update_preview)

    def _update_preview(self) -> None:
        self.refresh_job = None
        self.result, self.strength_map = render_local(*self.tensors, self.strength.get() / 100)
        self.enhanced.show(self.result)
        if self.show_map.get():
            self.map_pane.show(self.strength_map)

    def _finish_pending_preview(self) -> None:
        if self.refresh_job is not None:
            self.root.after_cancel(self.refresh_job)
            self._update_preview()

    def save_result(self, *, save_map: bool = False) -> None:
        if self.closing or self.result is None or self.pending is not None:
            return
        self._finish_pending_preview()
        image = self.strength_map if save_map else self.result
        kind, suffix = ("系数图", "strength") if save_map else ("增强图像", "enhanced")
        selected = filedialog.asksaveasfilename(
            parent=self.root, title=f"导出{kind}", initialfile=f"{self.path.stem}_{suffix}.png",
            defaultextension=".png", filetypes=[("PNG 图像", "*.png")],
        )
        if not selected:
            return
        target = Path(selected)
        if target.suffix.lower() != ".png" or target.resolve() == self.path.resolve():
            messagebox.showerror("无法导出", "请选择与原图不同的 PNG 文件名。", parent=self.root)
            return
        temporary = None
        try:
            with NamedTemporaryFile(dir=target.parent, prefix=".aqualab-", suffix=".png", delete=False) as stream:
                temporary = Path(stream.name)
                image.save(stream, format="PNG")
            temporary.replace(target)
        except OSError as error:
            messagebox.showerror("导出失败", str(error), parent=self.root)
        else:
            self.status.set(f"已导出 {target.name}  ·  {image.width} × {image.height}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def close(self) -> None:
        if self.closing:
            return
        self.closing = True
        self.open_button.state(["disabled"])
        self._enable_result(False)
        if self.refresh_job is not None:
            self.root.after_cancel(self.refresh_job)
            self.refresh_job = None
        if self.pending is not None and not self.pending.done() and not self.pending.cancel():
            self.status.set("正在等待后台推理结束，随后将关闭窗口…")
            return
        self._finish_close()

    def _finish_close(self) -> None:
        if self.poll_job is not None:
            self.root.after_cancel(self.poll_job)
            self.poll_job = None
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional image to open at launch.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    root = tk.Tk()
    window = AquaLabWindow(root, ROOT / "weights/wwe_uie_uieb.pth", args.device)
    if args.input is not None:
        root.after_idle(window.load_image, args.input)
    root.mainloop()


if __name__ == "__main__":
    main()
