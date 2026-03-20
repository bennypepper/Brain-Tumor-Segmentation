import os
import numpy as np
import cv2
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score, davies_bouldin_score
import streamlit as st
from PIL import Image
import io


st.set_page_config(
    page_title="Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide"
)


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
        
    def load_image_array(self, img_rgb):
        """Preprocess numpy RGB image array for segmentation."""
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if len(img_rgb.shape) > 2 else img_rgb
        original_shape = img_gray.shape
        
     
        pixel_intensity = img_gray.flatten().astype(np.float32)
        
        if self.use_texture:
       
            local_mean = cv2.blur(img_gray, (5, 5)).flatten().astype(np.float32)
            sq_img = img_gray.astype(np.float32) ** 2
            local_sq_mean = cv2.blur(sq_img, (5, 5)).flatten()
            local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
            
           
            pixel_data = np.vstack((pixel_intensity, local_mean, local_std))
        else:
           
            pixel_data = pixel_intensity.reshape((1, -1))
            
        return img_rgb, img_gray, pixel_data, original_shape
    
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
        """Convert cluster labels to color-coded segmented RGB image."""
        segmented_labels = cluster_labels.reshape(original_shape)
        segmented_color = np.zeros((original_shape[0], original_shape[1], 3), dtype=np.uint8)
        
    
        intensity_centers = self.cntr[:, 0]
        sorted_centers = np.argsort(intensity_centers)
        
      
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
                st.warning(f"Detailed metric calculation skipped: {e}")
        
        return metrics
    
    def process(self, img_rgb):
        """Full segmentation pipeline."""
        img_rgb, img_gray, pixel_data, original_shape = self.load_image_array(img_rgb)
        cluster_labels = self.segment(pixel_data)
        segmented_labels, segmented_color = self.create_segmented_image(cluster_labels, original_shape)
        metrics = self.evaluate(pixel_data, cluster_labels)
        
        return {
            'original': img_rgb,
            'gray': img_gray,
            'segmented_labels': segmented_labels,
            'segmented_color': segmented_color,
            'metrics': metrics
        }


def main():
    st.title("🧠 Brain Tumor Segmentation using FCM")
    st.markdown("Upload an MRI scan to automatically segment and highlight potential tumor regions using the **Fuzzy C-Means** algorithm.")

    st.sidebar.header("⚙️ Advanced Configuration")
    n_clusters = st.sidebar.slider("Number of Clusters (Tissue Types)", min_value=2, max_value=8, value=4, step=1)
    fuzziness = st.sidebar.slider("Fuzziness Parameter (m)", min_value=1.1, max_value=5.0, value=2.0, step=0.1)
    use_texture = st.sidebar.checkbox("Use Texture Features (Local Mean & Variance)", value=False)
    
    st.sidebar.markdown("---")

    with st.sidebar.expander("ℹ️ About this Program", expanded=False):
        st.markdown("""
        **Developer:** Benedict Pepper
        
        **About this Program:**
        This is an automated medical image analysis tool that applies the Fuzzy C-Means (FCM) soft clustering algorithm to segment brain MRI scans. By clustering pixel data into distinct tissue groups, this tool helps isolate potential tumor regions from healthy tissue, fluids, and background.
        
        **What is Fuzzy C-Means (FCM)?**
        Unlike traditional K-Means clustering where each pixel belongs strictly to ONE group, FCM allows a pixel to have a "degree of belonging" (membership) to multiple groups at the same time. This is particularly useful in medical imaging because tissue boundaries are often blurred and overlap (the partial volume effect).
        
        **The Objective Function (Formula):**
        The algorithm minimizes:
        $J_m = \sum_{i=1}^{N} \sum_{j=1}^{C} (u_{ij}^m) \cdot ||x_i - c_j||^2$
        
        Where:
        * $N$: total number of pixels.
        * $C$: total number of clusters.
        * $u_{ij}$: degree of membership of pixel $x_i$ in cluster $j$.
        * $m$: Fuzziness parameter.
        * $c_j$: center of the cluster.
        """)

    # Main area execution - User Input
    st.markdown("### 1. Upload MRI Scan")
    uploaded_file = st.file_uploader("Choose an MRI image file", type=["jpg", "jpeg", "png", "bmp", "tif"])

    if uploaded_file is not None:
     
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        st.markdown("### 2. Run Segmentation")
        run_btn = st.button("🚀 Run Segmentation", type="primary", use_container_width=True)

      
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption="Original MRI Image", use_container_width=True)
            
        with col2:
            if not run_btn:
                st.info("👈 Click **Run Segmentation** above to process this image. (You can adjust advanced settings in the sidebar first if desired)")
            else:
                with st.spinner("Processing... Applying Fuzzy C-Means (This might take a few seconds)"):
                    segmenter = FCMSegmenter(
                        n_clusters=n_clusters, 
                        fuzziness=fuzziness, 
                        use_texture=use_texture
                    )
                    results = segmenter.process(img_rgb)
                    
                st.image(results['segmented_color'], caption="Segmented Output", use_container_width=True)
                
      
        if run_btn:
            st.markdown("---")
            st.markdown("### 📊 Evaluation Metrics")
            m = results['metrics']
            
            metric_cols = st.columns(4)
            metric_cols[0].metric("FPC Score", f"{m.get('fpc', 0):.4f}", help="Fuzzy Partition Coefficient (closer to 1.0 is better)")
            metric_cols[1].metric("Iterations", f"{m.get('iterations', 0)}", help="Cycles taken to converge")
            metric_cols[2].metric("Objective Val", f"{m.get('objective_value', 0):.2e}")
            if 'silhouette' in m:
                metric_cols[3].metric("Silhouette Score", f"{m['silhouette']:.4f}", help="Cluster separation quality (closer to 1.0 is better)")

            st.markdown("### 💾 Export Results")
            
     
            seg_img_pil = Image.fromarray(results['segmented_color'])
            buf = io.BytesIO()
            seg_img_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Segmented Image",
                data=byte_im,
                file_name="segmented_tumor.png",
                mime="image/png"
            )
    else:
        st.info("👆 Please upload an MRI scan to get started.")

if __name__ == '__main__':
    main()