import csv
import pickle
import threading
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageTk


APP_DIR = __file__.replace("\\", "/").rsplit("/", 1)[0]


def app_path(filename):
    return APP_DIR + "/" + filename


class LogisticRegression:
    """Inference-only logistic regression — models are loaded from pickle, not trained here."""

    @staticmethod
    def _sigmoid(z):
        # Numerically stable sigmoid: split positive/negative to avoid overflow in exp()
        pos = z >= 0
        result = np.empty_like(z, dtype=float)
        result[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[~pos])
        result[~pos] = exp_z / (1.0 + exp_z)
        return result

    def _add_intercept(self, X):
        # Prepend a column of 1s for the bias term
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        X = self._add_intercept(X)
        probs_1 = self._sigmoid(X @ self.weights_)
        # Return [P(class=0), P(class=1)] per sample
        return np.column_stack([1 - probs_1, probs_1])


class LeNet5(nn.Module):
    """LeNet-5 CNN — only extract_features() is used at inference time.
    forward() and self.classifier exist so load_state_dict() can map all saved weights."""

    def __init__(self):
        super().__init__()
        # Convolutional backbone: 1×32×32 → 16×5×5
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),     # → 6 × 28 × 28
            nn.ReLU(),
            nn.AvgPool2d(2, 2),                 # → 6 × 14 × 14
            nn.Conv2d(6, 16, kernel_size=5),    # → 16 × 10 × 10
            nn.ReLU(),
            nn.AvgPool2d(2, 2),                 # → 16 × 5 × 5
        )
        # Fully-connected layers that produce the 84-d feature vector
        self.feature_fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        # Classification head — not used at inference, but needed for load_state_dict()
        self.classifier = nn.Linear(84, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        """Forward through backbone only — returns 84-d feature vector per image."""
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten: (batch, 16*5*5)
        x = self.feature_fc(x)
        return x  # shape: (batch, 84)


class DigitPredictorApp:
    CANVAS_SIZE = 280
    MNIST_SIZE = 28
    PADDED_SIZE = 32
    PREVIEW_SIZE = 112
    BG = "#15171c"
    PANEL = "#20232b"
    TEXT = "#f2f4f8"
    MUTED = "#aeb6c2"
    BAR_BG = "#343946"
    BAR_FILL = "#4cc9f0"

    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Live Predictor")
        self.root.resizable(False, False)
        self.root.configure(bg=self.BG)

        self.model = None
        self.lr_models = None
        if not self.load_models_or_exit():
            return

        self.image = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.undo_stack = []
        self.last_xy = None
        self.predict_after_id = None
        self.inference_lock = threading.Lock()
        self.request_id = 0
        self.latest_probs = None
        self.latest_pred = None
        self.total = 0
        self.correct = 0

        self.setup_styles()
        self.build_ui()
        self.update_stats()
        self.set_placeholder()

    def load_models_or_exit(self):
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = LeNet5()
            self.model.load_state_dict(torch.load(app_path("lenet5_mnist.pth"), map_location=self.device))
            # Freeze all parameters — no training, inference only
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            self.model = self.model.to(self.device)

            with open(app_path("lr_models.pkl"), "rb") as f:
                self.lr_models = pickle.load(f)
            return True
        except FileNotFoundError as exc:
            missing = exc.filename or "required model file"
            messagebox.showerror(
                "Missing model file",
                f"Could not find {missing} next to digit_gui.py.",
            )
        except Exception as exc:
            messagebox.showerror("Model loading failed", str(exc))
        self.root.destroy()
        return False

    def setup_styles(self):
        self.root.option_add("*Font", ("Segoe UI", 10))

    def build_ui(self):
        main = tk.Frame(self.root, bg=self.BG, padx=18, pady=18)
        main.grid(row=0, column=0)

        left = tk.Frame(main, bg=self.BG)
        left.grid(row=0, column=0, sticky="n")

        right = tk.Frame(main, bg=self.BG)
        right.grid(row=0, column=1, sticky="n", padx=(22, 0))

        # ── Drawing canvas ──
        self.canvas = tk.Canvas(
            left,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            bg="black",
            highlightthickness=2,
            highlightbackground="#3a4050",
            cursor="crosshair",
        )
        self.canvas.grid(row=0, column=0, columnspan=3)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.clear_btn = self.make_button(left, "Clear", self.clear_canvas, "#303540")
        self.clear_btn.grid(row=1, column=0, sticky="ew", pady=(12, 0), padx=(0, 6))

        self.undo_btn = self.make_button(left, "Undo", self.undo, "#303540")
        self.undo_btn.grid(row=1, column=1, sticky="ew", pady=(12, 0), padx=6)

        brush_frame = tk.Frame(left, bg=self.BG)
        brush_frame.grid(row=1, column=2, sticky="ew", pady=(12, 0), padx=(6, 0))
        tk.Label(brush_frame, text="Brush", bg=self.BG, fg=self.MUTED).grid(row=0, column=0)
        self.brush_size = tk.IntVar(value=16)
        self.brush_slider = tk.Scale(
            brush_frame,
            from_=6,
            to=34,
            orient="horizontal",
            variable=self.brush_size,
            bg=self.BG,
            fg=self.TEXT,
            troughcolor="#303540",
            highlightthickness=0,
            length=92,
            showvalue=True,
        )
        self.brush_slider.grid(row=1, column=0)

        # ── 28×28 preview of what the model actually sees ──
        preview_box = tk.Frame(left, bg=self.BG)
        preview_box.grid(row=2, column=0, columnspan=3, pady=(16, 0))
        tk.Label(preview_box, text="Model preview", bg=self.BG, fg=self.MUTED).grid(row=0, column=0)
        self.preview_label = tk.Label(
            preview_box,
            width=self.PREVIEW_SIZE,
            height=self.PREVIEW_SIZE,
            bg="black",
            bd=1,
            relief="solid",
        )
        self.preview_label.grid(row=1, column=0, pady=(6, 0))

        # ── Prediction display ──
        self.prediction_label = tk.Label(
            right,
            text="Draw a digit...",
            bg=self.BG,
            fg=self.TEXT,
            font=("Segoe UI", 40, "bold"),
            width=10,
            anchor="center",
        )
        self.prediction_label.grid(row=0, column=0, sticky="ew")

        # ── Per-digit probability bar chart ──
        self.chart = tk.Frame(right, bg=self.BG)
        self.chart.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        self.prob_bars = []
        self.prob_labels = []
        for digit in range(10):
            row = tk.Frame(self.chart, bg=self.BG)
            row.grid(row=digit, column=0, sticky="ew", pady=2)
            tk.Label(row, text=str(digit), bg=self.BG, fg=self.TEXT, width=2).grid(row=0, column=0)
            bar_canvas = tk.Canvas(
                row,
                width=220,
                height=16,
                bg=self.BAR_BG,
                highlightthickness=0,
            )
            bar_canvas.grid(row=0, column=1, padx=8)
            rect = bar_canvas.create_rectangle(0, 0, 0, 16, fill=self.BAR_FILL, width=0)
            pct = tk.Label(row, text="0.0%", bg=self.BG, fg=self.MUTED, width=7, anchor="e")
            pct.grid(row=0, column=2)
            self.prob_bars.append((bar_canvas, rect))
            self.prob_labels.append(pct)

        tk.Label(right, text="Top 3", bg=self.BG, fg=self.MUTED).grid(
            row=2, column=0, sticky="w", pady=(16, 0)
        )
        self.top3_label = tk.Label(
            right,
            text="-",
            bg=self.BG,
            fg=self.TEXT,
            justify="left",
            anchor="w",
            font=("Segoe UI", 12),
        )
        self.top3_label.grid(row=3, column=0, sticky="ew", pady=(6, 0))

        # ── Feedback panel for logging correct/wrong predictions ──
        feedback = tk.Frame(right, bg=self.PANEL, padx=12, pady=12)
        feedback.grid(row=4, column=0, sticky="ew", pady=(18, 0))
        tk.Label(feedback, text="Feedback", bg=self.PANEL, fg=self.TEXT, font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        self.correct_btn = self.make_button(feedback, "✓ Correct", self.mark_correct, "#137a4c")
        self.correct_btn.grid(row=1, column=0, sticky="ew", pady=(10, 0), padx=(0, 6))
        self.wrong_btn = self.make_button(feedback, "✗ Wrong", self.show_wrong_controls, "#9b1c31")
        self.wrong_btn.grid(row=1, column=1, sticky="ew", pady=(10, 0), padx=(6, 0))

        self.actual_label = tk.Label(feedback, text="What was the real digit?", bg=self.PANEL, fg=self.MUTED)
        self.actual_spin = tk.Spinbox(
            feedback,
            from_=0,
            to=9,
            width=4,
            bg="#111318",
            fg=self.TEXT,
            buttonbackground="#303540",
            insertbackground=self.TEXT,
            justify="center",
        )
        self.submit_btn = self.make_button(feedback, "Submit", self.submit_wrong, "#303540")

        self.stats_label = tk.Label(right, text="", bg=self.BG, fg=self.MUTED, anchor="center")
        self.stats_label.grid(row=5, column=0, sticky="ew", pady=(18, 0))

    def make_button(self, parent, text, command, bg):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=self.TEXT,
            activebackground=bg,
            activeforeground=self.TEXT,
            relief="flat",
            bd=0,
            padx=12,
            pady=8,
            cursor="hand2",
        )

    def on_press(self, event):
        self.save_undo_state()
        self.last_xy = (event.x, event.y)
        self.draw_point(event.x, event.y)
        self.schedule_prediction()

    def on_drag(self, event):
        if self.last_xy is None:
            self.last_xy = (event.x, event.y)
        x0, y0 = self.last_xy
        x1, y1 = event.x, event.y
        size = self.brush_size.get()
        # Draw on the visible canvas
        self.canvas.create_line(
            x0, y0, x1, y1,
            fill="white",
            width=size,
            capstyle=tk.ROUND,
            smooth=True,
        )
        # Mirror the stroke onto the PIL image (used for model input)
        self.draw.line((x0, y0, x1, y1), fill=255, width=size)
        r = size // 2
        self.draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=255)
        self.last_xy = (x1, y1)
        self.schedule_prediction()

    def on_release(self, _event):
        self.last_xy = None
        if self.predict_after_id is not None:
            self.root.after_cancel(self.predict_after_id)
            self.predict_after_id = None
        self.run_prediction()

    def draw_point(self, x, y):
        size = self.brush_size.get()
        r = size // 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse((x - r, y - r, x + r, y + r), fill=255)

    def save_undo_state(self):
        self.undo_stack.append(self.image.copy())
        if len(self.undo_stack) > 5:
            self.undo_stack.pop(0)

    def undo(self):
        if not self.undo_stack:
            return
        self.image = self.undo_stack.pop()
        self.draw = ImageDraw.Draw(self.image)
        self.redraw_canvas_from_image()
        self.run_prediction()

    def redraw_canvas_from_image(self):
        self.canvas.delete("all")
        self.canvas_photo = ImageTk.PhotoImage(self.image.convert("RGB"))
        self.canvas.create_image(0, 0, image=self.canvas_photo, anchor="nw")

    def clear_canvas(self):
        self.image = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.undo_stack.clear()
        self.canvas.delete("all")
        self.latest_probs = None
        self.latest_pred = None
        self.set_placeholder()
        self.update_preview(None)

    def schedule_prediction(self):
        # Debounce: wait 300ms after last stroke before running inference
        if self.predict_after_id is not None:
            self.root.after_cancel(self.predict_after_id)
        self.predict_after_id = self.root.after(300, self.run_prediction)

    def run_prediction(self):
        self.predict_after_id = None
        if self.is_blank():
            self.latest_probs = None
            self.latest_pred = None
            self.set_placeholder()
            self.update_preview(None)
            return

        # Run inference on a background thread to keep the UI responsive
        self.request_id += 1
        request_id = self.request_id
        image_copy = self.image.copy()
        thread = threading.Thread(
            target=self.inference_worker,
            args=(image_copy, request_id),
            daemon=True,
        )
        thread.start()

    def is_blank(self):
        return self.image.getbbox() is None

    def preprocess(self, image):
        """Convert 280×280 canvas drawing → normalized 1×1×32×32 tensor for LeNet-5."""
        bbox = image.getbbox()
        if bbox is None:
            return None, None

        # Downscale full canvas to 28×28 (MNIST native resolution)
        img_28 = image.resize((self.MNIST_SIZE, self.MNIST_SIZE), Image.Resampling.LANCZOS)
        arr = np.asarray(img_28, dtype=np.float32)

        # Pad 28×28 → 32×32 with 2px border (LeNet-5 expects 32×32 input)
        margin = (self.PADDED_SIZE - self.MNIST_SIZE) // 2
        padded = np.zeros((self.PADDED_SIZE, self.PADDED_SIZE), dtype=np.float32)
        padded[margin:margin + self.MNIST_SIZE, margin:margin + self.MNIST_SIZE] = arr

        # Normalize with MNIST global mean=0.1307 and std=0.3081
        arr_norm = padded / 255.0
        arr_norm = (arr_norm - 0.1307) / 0.3081
        # Add batch and channel dimensions: (32,32) → (1,1,32,32)
        tensor = torch.from_numpy(arr_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor, img_28

    def inference_worker(self, image, request_id):
        """Background thread: preprocess → LeNet-5 features → 10 LR models → softmax."""
        with self.inference_lock:
            try:
                tensor, preview = self.preprocess(image)
                if tensor is None:
                    self.root.after(0, self.set_placeholder)
                    return

                # Extract 84-d feature vector from LeNet-5 backbone
                with torch.no_grad():
                    features = self.model.extract_features(tensor).cpu().numpy()

                # Collect P(digit=k) from each of the 10 binary LR models
                probs = []
                for digit in range(10):
                    clf = self.lr_models[digit]
                    probs.append(float(clf.predict_proba(features)[:, 1][0]))

                # Apply softmax to convert raw probabilities into a proper distribution
                raw = np.asarray(probs, dtype=np.float64)
                exps = np.exp(raw - np.max(raw))  # subtract max for numerical stability
                softmax = exps / np.sum(exps)
                pred = int(np.argmax(softmax))

                # Post result back to the UI thread
                self.root.after(0, self.apply_prediction, request_id, pred, softmax, preview)
            except Exception as exc:
                self.root.after(0, self.show_inference_error, str(exc))

    def apply_prediction(self, request_id, pred, probs, preview):
        # Ignore stale results if a newer prediction was already requested
        if request_id != self.request_id:
            return
        self.latest_pred = pred
        self.latest_probs = probs
        self.prediction_label.configure(text=str(pred), font=("Segoe UI", 40, "bold"))
        self.update_bars(probs)
        self.update_top3(probs)
        self.update_preview(preview)

    def update_bars(self, probs):
        for digit, prob in enumerate(probs):
            width = int(220 * float(prob))
            canvas, rect = self.prob_bars[digit]
            canvas.coords(rect, 0, 0, width, 16)
            self.prob_labels[digit].configure(text=f"{prob * 100:.1f}%")

    def update_top3(self, probs):
        top = np.argsort(probs)[::-1][:3]
        lines = [f"{int(digit)}: {probs[digit] * 100:.2f}%" for digit in top]
        self.top3_label.configure(text="\n".join(lines))

    def update_preview(self, preview):
        if preview is None:
            blank = Image.new("L", (self.PREVIEW_SIZE, self.PREVIEW_SIZE), 0)
            self.preview_photo = ImageTk.PhotoImage(blank)
        else:
            scaled = preview.resize((self.PREVIEW_SIZE, self.PREVIEW_SIZE), Image.Resampling.NEAREST)
            self.preview_photo = ImageTk.PhotoImage(scaled)
        self.preview_label.configure(image=self.preview_photo)

    def set_placeholder(self):
        self.prediction_label.configure(text="Draw a digit...", font=("Segoe UI", 24, "bold"))
        for digit in range(10):
            canvas, rect = self.prob_bars[digit]
            canvas.coords(rect, 0, 0, 0, 16)
            self.prob_labels[digit].configure(text="0.0%")
        self.top3_label.configure(text="-")

    def show_inference_error(self, message):
        messagebox.showerror("Inference failed", message)

    def show_wrong_controls(self):
        if self.latest_pred is None or self.latest_probs is None:
            return
        self.actual_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self.actual_spin.grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.submit_btn.grid(row=3, column=1, sticky="ew", pady=(8, 0), padx=(8, 0))

    def hide_wrong_controls(self):
        self.actual_label.grid_remove()
        self.actual_spin.grid_remove()
        self.submit_btn.grid_remove()

    def mark_correct(self):
        if self.latest_pred is None or self.latest_probs is None:
            return
        self.log_feedback(self.latest_pred, self.latest_pred, True)
        self.total += 1
        self.correct += 1
        self.update_stats()
        self.hide_wrong_controls()
        self.root.after(500, self.clear_canvas)

    def submit_wrong(self):
        if self.latest_pred is None or self.latest_probs is None:
            return
        actual = int(self.actual_spin.get())
        self.log_feedback(self.latest_pred, actual, False)
        self.total += 1
        self.update_stats()
        self.hide_wrong_controls()
        self.root.after(500, self.clear_canvas)

    def log_feedback(self, predicted, actual, is_correct):
        """Append prediction result to feedback_log.csv for tracking accuracy over time."""
        path = app_path("feedback_log.csv")
        try:
            with open(path, "r", newline="", encoding="utf-8"):
                file_exists = True
        except FileNotFoundError:
            file_exists = False
        probs = "" if self.latest_probs is None else " ".join(f"{p:.8f}" for p in self.latest_probs)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "predicted", "actual", "correct", "softmax_probabilities"])
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    predicted,
                    actual,
                    is_correct,
                    probs,
                ]
            )

    def update_stats(self):
        accuracy = 0.0 if self.total == 0 else (self.correct / self.total) * 100
        self.stats_label.configure(
            text=f"Total: {self.total} | Correct: {self.correct} | Accuracy: {accuracy:.1f}%"
        )


def main():
    root = tk.Tk()
    app = DigitPredictorApp(root)
    if app.model is not None and app.lr_models is not None:
        root.mainloop()


if __name__ == "__main__":
    main()
