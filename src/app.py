import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DEVICE, OUTPUT_DIR, STYLE_IMAGES
from utils import load_image, tensor_to_image
from style_transfer import run_style_transfer
from classical import run_classical, load_image_pil


class StyleTransferApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎨 Art Style Transfer App")
        self.root.geometry("1150x780")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(True, True)

        # State
        self.content_path  = None
        self.style_path    = None
        self.result_pil    = None
        self.processing    = False

        # Tk variables
        self.method_var          = tk.StringVar(value="neural")
        self.neural_style_var    = tk.StringVar(value="vangogh")
        self.classical_method_var = tk.StringVar(value="reinhard")

        # Image references (prevent garbage collection)
        self.content_photo = None
        self.style_photo   = None
        self.result_photo  = None

        self.build_ui()
        self.on_neural_style_change()  # Load default style image

    # ── UI Builder ──────────────────────────────────────────────

    def build_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg="#1e1e2e")
        title_frame.pack(fill="x", pady=10)

        tk.Label(
            title_frame,
            text="🎨 Art Style Transfer App",
            font=("Helvetica", 22, "bold"),
            bg="#1e1e2e", fg="#cdd6f4"
        ).pack()

        tk.Label(
            title_frame,
            text="Neural VGG19  •  Classical Methods  •  RTX 4060 GPU",
            font=("Helvetica", 10),
            bg="#1e1e2e", fg="#6c7086"
        ).pack()

        # Top row
        top_frame = tk.Frame(self.root, bg="#1e1e2e")
        top_frame.pack(fill="x", padx=20, pady=5)

        self.build_image_panel(top_frame, "📸 Content Image", "content")
        self.build_settings_panel(top_frame)
        self.build_image_panel(top_frame, "🖼️ Style Image",   "style")

        # Progress
        self.build_progress_bar()

        # Result
        self.build_result_panel()

    def build_image_panel(self, parent, title, panel_type):
        frame = tk.LabelFrame(
            parent, text=title,
            font=("Helvetica", 10, "bold"),
            bg="#313244", fg="#cdd6f4",
            relief="flat", padx=10, pady=10
        )
        frame.pack(side="left", padx=10, pady=5, fill="y")

        canvas = tk.Canvas(
            frame, width=220, height=220,
            bg="#1e1e2e", highlightthickness=1,
            highlightbackground="#45475a"
        )
        canvas.pack(pady=5)
        canvas.create_text(
            110, 110, text="No image\nselected",
            fill="#6c7086", font=("Helvetica", 11),
            justify="center", tags="placeholder"
        )

        btn = tk.Button(
            frame, text="📂 Browse",
            font=("Helvetica", 10, "bold"),
            bg="#89b4fa", fg="#1e1e2e",
            activebackground="#74c7ec",
            relief="flat", cursor="hand2",
            padx=15, pady=6,
            command=lambda: self.browse_image(panel_type, canvas)
        )
        btn.pack(pady=5)

        path_label = tk.Label(
            frame, text="No file selected",
            font=("Helvetica", 8),
            bg="#313244", fg="#6c7086",
            wraplength=220
        )
        path_label.pack()

        if panel_type == "content":
            self.content_canvas     = canvas
            self.content_path_label = path_label
        else:
            self.style_canvas       = canvas
            self.style_path_label   = path_label
            self.style_browse_btn   = btn

    def build_settings_panel(self, parent):
        frame = tk.LabelFrame(
            parent, text="⚙️ Settings",
            font=("Helvetica", 10, "bold"),
            bg="#313244", fg="#cdd6f4",
            relief="flat", padx=15, pady=15
        )
        frame.pack(side="left", padx=10, pady=5, fill="both", expand=True)

        # ── Method ──
        tk.Label(
            frame, text="Method:",
            font=("Helvetica", 11, "bold"),
            bg="#313244", fg="#cdd6f4"
        ).pack(anchor="w", pady=(0, 4))

        for text, val in [("🧠 Neural (VGG19)", "neural"),
                          ("🖌️ Classical",       "classical")]:
            tk.Radiobutton(
                frame, text=text,
                variable=self.method_var, value=val,
                font=("Helvetica", 10),
                bg="#313244", fg="#cdd6f4",
                selectcolor="#45475a",
                activebackground="#313244",
                activeforeground="#cdd6f4",
                cursor="hand2",
                command=self.on_method_change
            ).pack(anchor="w", pady=1)

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=10)

        # ── Neural options ──
        self.neural_frame = tk.Frame(frame, bg="#313244")
        self.neural_frame.pack(fill="x")

        tk.Label(
            self.neural_frame, text="Art Style:",
            font=("Helvetica", 11, "bold"),
            bg="#313244", fg="#cdd6f4"
        ).pack(anchor="w", pady=(0, 4))

        for text, val in [
            ("🌀 Van Gogh — Starry Night", "vangogh"),
            ("🌸 Monet — Water Lilies",    "monet"),
            ("🎨 Picasso — Cubism",        "picasso"),
            ("💥 Warhol — Pop Art",        "warhol"),
            ("📂 Custom (browse style)",   "custom"),
        ]:
            tk.Radiobutton(
                self.neural_frame, text=text,
                variable=self.neural_style_var, value=val,
                font=("Helvetica", 10),
                bg="#313244", fg="#cdd6f4",
                selectcolor="#45475a",
                activebackground="#313244",
                activeforeground="#cdd6f4",
                cursor="hand2",
                command=self.on_neural_style_change
            ).pack(anchor="w", pady=1)

        # ── Classical options ──
        self.classical_frame = tk.Frame(frame, bg="#313244")

        tk.Label(
            self.classical_frame, text="Classical Method:",
            font=("Helvetica", 11, "bold"),
            bg="#313244", fg="#cdd6f4"
        ).pack(anchor="w", pady=(0, 4))

        for text, val in [
            ("🎨 Reinhard Color Transfer", "reinhard"),
            ("📊 Histogram Matching",      "histogram"),
            ("✏️  Pencil Sketch",          "sketch"),
        ]:
            tk.Radiobutton(
                self.classical_frame, text=text,
                variable=self.classical_method_var, value=val,
                font=("Helvetica", 10),
                bg="#313244", fg="#cdd6f4",
                selectcolor="#45475a",
                activebackground="#313244",
                activeforeground="#cdd6f4",
                cursor="hand2",
                command=self.on_classical_method_change
            ).pack(anchor="w", pady=1)

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=10)

        # ── Process button ──
        self.process_btn = tk.Button(
            frame, text="🚀 Process",
            font=("Helvetica", 13, "bold"),
            bg="#a6e3a1", fg="#1e1e2e",
            activebackground="#94e2d5",
            relief="flat", cursor="hand2",
            padx=20, pady=10,
            command=self.process
        )
        self.process_btn.pack(fill="x", pady=5)

        self.status_label = tk.Label(
            frame, text="Ready ✅",
            font=("Helvetica", 9),
            bg="#313244", fg="#a6e3a1"
        )
        self.status_label.pack(pady=2)

        tk.Label(
            frame, text=f"Device: {DEVICE}",
            font=("Helvetica", 9),
            bg="#313244", fg="#6c7086"
        ).pack(pady=2)

    def build_progress_bar(self):
        frame = tk.Frame(self.root, bg="#1e1e2e")
        frame.pack(fill="x", padx=20, pady=4)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate"
        )
        self.progress_bar.pack(fill="x")

        self.progress_label = tk.Label(
            frame, text="",
            font=("Helvetica", 8),
            bg="#1e1e2e", fg="#6c7086"
        )
        self.progress_label.pack()

    def build_result_panel(self):
        frame = tk.LabelFrame(
            self.root, text="✅ Result",
            font=("Helvetica", 10, "bold"),
            bg="#313244", fg="#cdd6f4",
            relief="flat", padx=10, pady=10
        )
        frame.pack(fill="both", expand=True, padx=20, pady=5)

        self.result_canvas = tk.Canvas(
            frame, bg="#1e1e2e",
            highlightthickness=1,
            highlightbackground="#45475a",
            height=220
        )
        self.result_canvas.pack(fill="both", expand=True, pady=5)
        self.result_canvas.create_text(
            400, 110,
            text="Result will appear here after processing",
            fill="#6c7086", font=("Helvetica", 12),
            tags="placeholder"
        )

        self.save_btn = tk.Button(
            frame, text="💾 Save Result",
            font=("Helvetica", 10, "bold"),
            bg="#f38ba8", fg="#1e1e2e",
            activebackground="#fab387",
            relief="flat", cursor="hand2",
            padx=15, pady=6,
            state="disabled",
            command=self.save_result
        )
        self.save_btn.pack(pady=5)

    # ── Event handlers ──────────────────────────────────────────

    def on_method_change(self):
        if self.method_var.get() == "neural":
            self.classical_frame.pack_forget()
            self.neural_frame.pack(fill="x")
            self.on_neural_style_change()
        else:
            self.neural_frame.pack_forget()
            self.classical_frame.pack(fill="x")
            self.on_classical_method_change()

    def on_neural_style_change(self):
        style = self.neural_style_var.get()
        if style == "custom":
            self.style_browse_btn.config(state="normal")
            self.style_path = None
            self.style_canvas.delete("all")
            self.style_canvas.create_text(
                110, 110, text="Browse your\nstyle image",
                fill="#89b4fa", font=("Helvetica", 11),
                justify="center"
            )
            self.style_path_label.config(text="Browse a style image")
        else:
            self.style_browse_btn.config(state="disabled")
            preset = STYLE_IMAGES.get(style)
            if preset and os.path.exists(preset):
                self.style_path = preset
                self.style_path_label.config(text=os.path.basename(preset))
                self.display_preview(self.style_canvas, preset, "style")
            else:
                self.style_canvas.delete("all")
                self.style_canvas.create_text(
                    110, 110, text="Image not found\ncheck images/style/",
                    fill="#f38ba8", font=("Helvetica", 10),
                    justify="center"
                )

    def on_classical_method_change(self):
        if self.classical_method_var.get() == "sketch":
            self.style_browse_btn.config(state="disabled")
            self.style_canvas.delete("all")
            self.style_canvas.create_text(
                110, 110, text="Not needed\nfor sketch ✏️",
                fill="#6c7086", font=("Helvetica", 11),
                justify="center"
            )
        else:
            self.style_browse_btn.config(state="normal")
            if not self.style_path:
                self.style_canvas.delete("all")
                self.style_canvas.create_text(
                    110, 110, text="Browse a\nstyle image",
                    fill="#89b4fa", font=("Helvetica", 11),
                    justify="center"
                )

    # ── Image handling ──────────────────────────────────────────

    def browse_image(self, panel_type, canvas):
        path = filedialog.askopenfilename(
            title=f"Select {'Content' if panel_type == 'content' else 'Style'} Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path:
            return

        if panel_type == "content":
            self.content_path = path
            self.content_path_label.config(text=os.path.basename(path))
            self.display_preview(canvas, path, "content")
        else:
            self.style_path = path
            self.style_path_label.config(text=os.path.basename(path))
            self.display_preview(canvas, path, "style")

    def display_preview(self, canvas, path, ref_name, size=(220, 220)):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail(size, Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            canvas.delete("all")
            canvas.create_image(size[0] // 2, size[1] // 2,
                                anchor="center", image=photo)
            if ref_name == "content":
                self.content_photo = photo
            elif ref_name == "style":
                self.style_photo = photo
            elif ref_name == "result":
                self.result_photo = photo
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image:\n{e}")

    # ── Processing ──────────────────────────────────────────────

    def validate(self):
        if not self.content_path or not os.path.exists(self.content_path):
            messagebox.showwarning("Missing Input", "Please select a content image!")
            return False

        method = self.method_var.get()
        if method == "neural":
            if not self.style_path or not os.path.exists(self.style_path):
                messagebox.showwarning("Missing Input",
                                       "Style image not found!\nCheck images/style/ folder.")
                return False

        if method == "classical":
            if self.classical_method_var.get() != "sketch":
                if not self.style_path or not os.path.exists(self.style_path):
                    messagebox.showwarning("Missing Input",
                                           "Please browse a style image!")
                    return False
        return True

    def process(self):
        if self.processing:
            return
        if not self.validate():
            return

        self.processing = True
        self.process_btn.config(state="disabled", text="⏳ Processing...")
        self.save_btn.config(state="disabled")
        self.progress_var.set(0)
        self.result_canvas.delete("all")
        self.result_canvas.create_text(
            400, 110, text="Processing... please wait ⏳",
            fill="#89b4fa", font=("Helvetica", 12)
        )

        thread = threading.Thread(target=self.run_processing, daemon=True)
        thread.start()

    def run_processing(self):
        try:
            if self.method_var.get() == "neural":
                self.run_neural()
            else:
                self.run_classical()
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, self.reset_ui)

    def run_neural(self):
        style_name = self.neural_style_var.get()

        self.update_status("📸 Loading images...")
        self.update_progress(5)

        content_tensor = load_image(self.content_path)
        style_tensor   = load_image(self.style_path)

        self.update_status("🧠 Running Neural Style Transfer...")
        self.update_progress(10)

        from config import NUM_STEPS

        def progress_callback(step, total):
            pct = 10 + int((step / total) * 85)
            self.update_progress(pct)
            self.update_status(f"🧠 Neural: Step {step}/{total}")

        result_tensor = run_style_transfer(
            content_tensor, style_tensor,
            style_name=style_name,
            progress_callback=progress_callback
        )

        self.result_pil = tensor_to_image(result_tensor)
        self.update_progress(100)
        self.update_status("✅ Done!")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out = os.path.join(OUTPUT_DIR, f"result_{style_name}.jpg")
        self.result_pil.save(out)
        print(f"Saved → {out}")

        self.root.after(0, self.show_result)

    def run_classical(self):
        method_name = self.classical_method_var.get()

        self.update_status("📸 Loading images...")
        self.update_progress(20)

        content_pil = load_image_pil(self.content_path)
        style_pil   = load_image_pil(self.style_path) if self.style_path else None

        self.update_status(f"🖌️ Running {method_name}...")
        self.update_progress(50)

        self.result_pil = run_classical(method_name, content_pil, style_pil)

        self.update_progress(100)
        self.update_status("✅ Done!")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out = os.path.join(OUTPUT_DIR, f"result_classical_{method_name}.jpg")
        self.result_pil.save(out)
        print(f"Saved → {out}")

        self.root.after(0, self.show_result)

    def show_result(self):
        if self.result_pil is None:
            return

        self.result_canvas.update()
        w = self.result_canvas.winfo_width()  or 800
        h = self.result_canvas.winfo_height() or 220

        img = self.result_pil.copy()
        img.thumbnail((w - 20, h - 20), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)

        self.result_canvas.delete("all")
        self.result_canvas.create_image(w // 2, h // 2,
                                        anchor="center", image=photo)
        self.result_photo = photo
        self.save_btn.config(state="normal")
        self.reset_ui()

    def save_result(self):
        if not self.result_pil:
            return
        path = filedialog.asksaveasfilename(
            title="Save Result",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )
        if path:
            self.result_pil.save(path)
            messagebox.showinfo("Saved!", f"Result saved to:\n{path}")

    # ── Helpers ─────────────────────────────────────────────────

    def update_status(self, text):
        self.root.after(0, lambda: self.status_label.config(text=text))

    def update_progress(self, value):
        self.root.after(0, lambda: self.progress_var.set(value))
        self.root.after(0, lambda: self.progress_label.config(
            text=f"{int(value)}%" if value > 0 else ""
        ))

    def reset_ui(self):
        self.processing = False
        self.process_btn.config(state="normal", text="🚀 Process")


if __name__ == "__main__":
    root = tk.Tk()
    app  = StyleTransferApp(root)
    root.mainloop()