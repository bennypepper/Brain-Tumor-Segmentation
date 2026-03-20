import os
import sys
import threading
import numpy as np
import cv2
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score, davies_bouldin_score


import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk


class FCMSegmenter:
    """Fuzzy C-Means based MRI tumor segmentation with optional texture features."""
    
    def __init__(self, n_clusters=4, fuzziness=2.0, max_iter=1000, error=0.005, use_texture=False):
        self.n_clusters = n_clusters
        self.fuzziness = fuzziness
        self.max_iter = max_iter
        self.error = error
        self.use_texture = use_texture
        self.cntr = None
        self.u = None
        self.fpc = None
        self.iterations = None
        self.objective_value = None
        
    def load_image(self, image_path):
        """Load and preprocess image for segmentation."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
        original_shape = img_gray.shape
        
        # Base Intensity Feature
        pixel_intensity = img_gray.flatten().astype(np.float32)
        
        if self.use_texture:
            # Texture Matrix Features: Local Mean and Standard Deviation (Variance)
            
            local_mean = cv2.blur(img_gray, (5, 5)).flatten().astype(np.float32)
            sq_img = img_gray.astype(np.float32) ** 2
            local_sq_mean = cv2.blur(sq_img, (5, 5)).flatten()
            local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
            
            # Stack features: [Intensity, Local Mean, Local Std Dev]
            pixel_data = np.vstack((pixel_intensity, local_mean, local_std))
        else:
   
            pixel_data = pixel_intensity.reshape((1, -1))
            
        return img, img_gray, pixel_data, original_shape
    
    def segment(self, pixel_data):
        """Apply Fuzzy C-Means clustering to pixel data."""
        cntr, u, _, _, jm, p, fpc = fuzz.cluster.cmeans(
            data=pixel_data,
            c=self.n_clusters,
            m=self.fuzziness,
            error=self.error,
            maxiter=self.max_iter,
            init=None
        )
        
        self.cntr = cntr
        self.u = u
        self.fpc = fpc
        self.iterations = p
        self.objective_value = jm[-1]
        
        return np.argmax(u, axis=0)
    
    def create_segmented_image(self, cluster_labels, original_shape):
        """Convert cluster labels to color-coded segmented image."""
        segmented_labels = cluster_labels.reshape(original_shape)
        segmented_color = np.zeros((original_shape[0], original_shape[1], 3), dtype=np.uint8)
        
        # Sort centers by their primary feature (Intensity) to ensure consistent coloring
        intensity_centers = self.cntr[:, 0]
        sorted_centers = np.argsort(intensity_centers)
        
        # Dynamic Color map (darkest to brightest tissues)
        color_map = {
            sorted_centers[0]: [0, 0, 0],       # Background (Black)
            sorted_centers[1]: [85, 85, 85],    # Healthy tissue (Gray)
            sorted_centers[2]: [0, 255, 0],     # Fluid/Edema (Green)
            sorted_centers[3] if self.n_clusters > 3 else -1: [255, 0, 0] # Tumor (Red)
        }
        
        for i in range(self.n_clusters):
            mask = segmented_labels == i
            color = color_map.get(i, [255, 255, 255])
            segmented_color[mask] = color
        
        return segmented_labels, segmented_color
    
    def evaluate(self, pixel_data, cluster_labels):
        """Compute clustering quality metrics safely without freezing."""
        labels_flat = cluster_labels.flatten()
        data_t = pixel_data.T 
        
        metrics = {
            'fpc': self.fpc,
            'iterations': self.iterations,
            'objective_value': self.objective_value
        }
        
        if len(np.unique(labels_flat)) > 1:
            try:
                
                sample_size = min(10000, len(labels_flat))
                indices = np.random.choice(len(labels_flat), sample_size, replace=False)
                
                metrics['silhouette'] = silhouette_score(data_t[indices], labels_flat[indices])
                metrics['davies_bouldin'] = davies_bouldin_score(data_t[indices], labels_flat[indices])
            except Exception as e:
                print(f"Detailed metric calculation skipped: {e}")
        
        return metrics
    
    def process(self, image_path):
        """Full segmentation pipeline."""
        img_bgr, img_gray, pixel_data, original_shape = self.load_image(image_path)
        cluster_labels = self.segment(pixel_data)
        segmented_labels, segmented_color = self.create_segmented_image(cluster_labels, original_shape)
        metrics = self.evaluate(pixel_data, cluster_labels)
        
        return {
            'original': cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            'gray': img_gray,
            'segmented_labels': segmented_labels,
            'segmented_color': segmented_color,
            'metrics': metrics
        }
    
    def save_results(self, results, output_dir='output'):
        """Save segmentation results to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(
            os.path.join(output_dir, 'segmented_color.png'),
            cv2.cvtColor(results['segmented_color'], cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            os.path.join(output_dir, 'segmented_labels.png'),
            results['segmented_labels'].astype(np.uint8)
        )
        
        with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
            f.write("FCM Segmentation Metrics\n")
            f.write(f"FPC: {results['metrics'].get('fpc', 'N/A'):.4f}\n")
            f.write(f"Iterations: {results['metrics'].get('iterations', 'N/A')}\n")
            f.write(f"Objective Value: {results['metrics'].get('objective_value', 'N/A'):.4f}\n")
            if 'silhouette' in results['metrics']:
                f.write(f"Silhouette Score: {results['metrics']['silhouette']:.4f}\n")
            if 'davies_bouldin' in results['metrics']:
                f.write(f"Davies-Bouldin Index: {results['metrics']['davies_bouldin']:.4f}\n")


class FCMApp:
    """Tkinter Application for FCM Segmentation."""
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Segmentation (FCM)")
        self.root.geometry("950x650")
        
        self.image_path = None
        self.results = None
        self.use_texture = tk.BooleanVar(value=False)
        self.n_clusters = tk.IntVar(value=4)
        self.fuzziness = tk.DoubleVar(value=2.0)
        
        self._setup_ui()
        
    def _setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- Left Panel (Controls) ---
        control_frame = ttk.LabelFrame(main_frame, text="Settings & Controls", padding=15)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        ttk.Button(control_frame, text="Load MRI Image", command=self.load_image).pack(fill=tk.X, pady=5)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Label(control_frame, text="Clusters (Tissue Types):").pack(anchor=tk.W)
        ttk.Spinbox(control_frame, from_=2, to=8, textvariable=self.n_clusters, width=10).pack(anchor=tk.W, pady=2)
        
        ttk.Label(control_frame, text="Fuzziness (m):").pack(anchor=tk.W)
        ttk.Entry(control_frame, textvariable=self.fuzziness, width=12).pack(anchor=tk.W, pady=2)
        
        ttk.Checkbutton(control_frame, text="Use Texture Features\n(Local Mean & Var)", variable=self.use_texture).pack(anchor=tk.W, pady=10)
        
        self.btn_run = ttk.Button(control_frame, text="Run Segmentation", command=self.run_segmentation)
        self.btn_run.pack(fill=tk.X, pady=15)
        
        self.btn_save = ttk.Button(control_frame, text="Save Results", state=tk.DISABLED, command=self.save_results)
        self.btn_save.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="Evaluation Metrics:").pack(anchor=tk.W, pady=(15, 0))
        self.metrics_text = tk.Text(control_frame, height=8, width=25, state=tk.DISABLED, bg="#f9f9f9")
        self.metrics_text.pack(fill=tk.X, pady=5)
        
       
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Button(control_frame, text="ℹ️ About / Info", command=self.show_about).pack(fill=tk.X, pady=5)

       
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
   
        self.lbl_orig = ttk.Label(image_frame, text="Original Image\n(Please load an image)", anchor=tk.CENTER, relief=tk.SUNKEN)
        self.lbl_orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.lbl_seg = ttk.Label(image_frame, text="Segmented Output", anchor=tk.CENTER, relief=tk.SUNKEN)
        self.lbl_seg.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

    def show_about(self):
        """Displays an informational dialog about the program and FCM."""
        about_win = tk.Toplevel(self.root)
        about_win.title("About this Program")
        about_win.geometry("500x550")
        about_win.resizable(False, False)
        about_win.transient(self.root) 
        about_win.grab_set()           
        
      
        txt = tk.Text(about_win, wrap=tk.WORD, padx=15, pady=15, bg=self.root.cget('bg'), font=("Arial", 10))
        txt.pack(fill=tk.BOTH, expand=True)
        
        about_content = """🧠 Brain Tumor Segmentation Tool

Developer:
Benedict Pepper

About this Program:
This is an automated medical image analysis tool that applies the Fuzzy C-Means (FCM) soft clustering algorithm to segment brain MRI scans. By clustering pixel data into distinct tissue groups, this tool helps isolate potential tumor regions from healthy tissue, fluids, and background.

What is Fuzzy C-Means (FCM)?
Unlike traditional K-Means clustering where each pixel belongs strictly to ONE group, FCM allows a pixel to have a "degree of belonging" (membership) to multiple groups at the same time. This is particularly useful in medical imaging because tissue boundaries are often blurred and overlap (the partial volume effect).

The Objective Function (Formula):
The algorithm works by minimizing the following objective function:

    J_m = Σ (from i=1 to N) Σ (from j=1 to C)  (u_ij^m) * ||x_i - c_j||²

Where:
• N is the total number of pixels.
• C is the total number of clusters.
• u_ij is the degree of membership of pixel x_i in cluster j.
• m is the Fuzziness parameter (usually 2.0).
• c_j is the center of the cluster.
• ||x_i - c_j||² is the squared Euclidean distance between the pixel and the cluster center.

Evaluation Metrics:
• FPC (Fuzzy Partition Coefficient): Measures how cleanly the data is clustered. Values closer to 1.0 indicate exceptionally well-defined and compact clusters.
"""
        txt.insert(tk.END, about_content)
        txt.config(state=tk.DISABLED) # Make text read-only
        
        ttk.Button(about_win, text="Close", command=about_win.destroy).pack(pady=10)

    def array_to_imagetk(self, arr_rgb, max_width=400):
        """Convert numpy RGB array to Tkinter compatible format."""
        h, w = arr_rgb.shape[:2]
        ratio = max_width / float(w)
        new_h = int(h * ratio)
        resized = cv2.resize(arr_rgb, (max_width, new_h))
        img = Image.fromarray(resized)
        return ImageTk.PhotoImage(image=img)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Select MRI Scan",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif")]
        )
        if self.image_path:
            # Preview Original
            img_bgr = cv2.imread(self.image_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            self.orig_tk = self.array_to_imagetk(img_rgb)
            self.lbl_orig.config(image=self.orig_tk, text="")
            
            # Reset Segmentation label
            self.lbl_seg.config(image='', text="Ready for segmentation...")
            self.btn_save.config(state=tk.DISABLED)

    def run_segmentation(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please select an MRI image first.")
            return
            
        self.btn_run.config(state=tk.DISABLED, text="Processing...")
        self.root.update()
        
        # Run segmentation in background to avoid freezing the GUI
        def processing_task():
            try:
                segmenter = FCMSegmenter(
                    n_clusters=self.n_clusters.get(),
                    fuzziness=self.fuzziness.get(),
                    use_texture=self.use_texture.get()
                )
                self.results = segmenter.process(self.image_path)
                
                # Update UI elements back on the main thread safely
                self.root.after(0, self.update_ui_post_segmentation)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Segmentation Failed:\n{str(e)}"))
            finally:
                self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL, text="Run Segmentation"))
                
        threading.Thread(target=processing_task, daemon=True).start()

    def update_ui_post_segmentation(self):
        # Update Segmented Image View
        self.seg_tk = self.array_to_imagetk(self.results['segmented_color'])
        self.lbl_seg.config(image=self.seg_tk)
        self.btn_save.config(state=tk.NORMAL)
        
        # Update Metrics view
        m = self.results['metrics']
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, f"FPC: {m.get('fpc', 0):.4f}\n")
        self.metrics_text.insert(tk.END, f"Iterations: {m.get('iterations', 0)}\n")
        self.metrics_text.insert(tk.END, f"Objective: {m.get('objective_value', 0):.2f}\n")
        if 'silhouette' in m:
            self.metrics_text.insert(tk.END, f"Silhouette: {m['silhouette']:.4f}\n")
        self.metrics_text.config(state=tk.DISABLED)

    def save_results(self):
        if not self.results:
            return
        out_dir = filedialog.askdirectory(title="Select Output Directory")
        if out_dir:
            segmenter = FCMSegmenter()
            segmenter.save_results(self.results, out_dir)
            messagebox.showinfo("Success", f"Results successfully saved to:\n{out_dir}")


def main():
    import argparse
    
    # Dual-Mode Support: If arguments are passed, run CLI, otherwise run GUI.
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='FCM-based MRI Tumor Segmentation')
        parser.add_argument('--input', type=str, required=True, help='Path to input MRI image')
        parser.add_argument('--output', type=str, default='output', help='Output directory')
        parser.add_argument('--clusters', type=int, default=4, help='Number of clusters')
        parser.add_argument('--fuzziness', type=float, default=2.0, help='Fuzziness parameter')
        args = parser.parse_args()
        
        segmenter = FCMSegmenter(
            n_clusters=args.clusters,
            fuzziness=args.fuzziness,
            use_texture=False
        )
        
        print(f"Processing: {args.input}")
        results = segmenter.process(args.input)
        
        print("\nSegmentation complete.")
        print(f"FPC: {results['metrics']['fpc']:.4f}")
        print(f"Iterations: {results['metrics']['iterations']}")
        
        segmenter.save_results(results, args.output)
        print(f"Results saved to: {args.output}/")
    else:
        
        root = tk.Tk()
        app = FCMApp(root)
        root.mainloop()


if __name__ == '__main__':
    main()