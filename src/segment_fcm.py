import os
import numpy as np
import cv2
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score, davies_bouldin_score


class FCMSegmenter:
    """Fuzzy C-Means based MRI tumor segmentation."""
    
    def __init__(self, n_clusters=4, fuzziness=2.0, max_iter=1000, error=0.005):
        self.n_clusters = n_clusters
        self.fuzziness = fuzziness
        self.max_iter = max_iter
        self.error = error
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
        pixel_data = img_gray.flatten().astype(np.float32).reshape((1, -1))
        
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
        
        sorted_centers = np.argsort(self.cntr.flatten())
        color_map = {
            sorted_centers[0]: [0, 0, 0],
            sorted_centers[1]: [85, 85, 85],
            sorted_centers[2]: [0, 255, 0],
            sorted_centers[3]: [255, 0, 0]
        }
        
        for i in range(self.n_clusters):
            mask = segmented_labels == i
            color = color_map.get(i, [255, 255, 255])
            segmented_color[mask] = color
        
        return segmented_labels, segmented_color
    
    def evaluate(self, pixel_data, cluster_labels):
        """Compute clustering quality metrics."""
        labels_flat = cluster_labels.flatten()
        data_flat = pixel_data.flatten()
        
        metrics = {
            'fpc': self.fpc,
            'iterations': self.iterations,
            'objective_value': self.objective_value
        }
        
        if len(np.unique(labels_flat)) > 1:
            try:
                metrics['silhouette'] = silhouette_score(data_flat.reshape(-1, 1), labels_flat)
                metrics['davies_bouldin'] = davies_bouldin_score(data_flat.reshape(-1, 1), labels_flat)
            except Exception:
                pass
        
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


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='FCM-based MRI Tumor Segmentation')
    parser.add_argument('--input', type=str, required=True, help='Path to input MRI image')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--clusters', type=int, default=4, help='Number of clusters')
    parser.add_argument('--fuzziness', type=float, default=2.0, help='Fuzziness parameter')
    args = parser.parse_args()
    
    segmenter = FCMSegmenter(
        n_clusters=args.clusters,
        fuzziness=args.fuzziness
    )
    
    print(f"Processing: {args.input}")
    results = segmenter.process(args.input)
    
    print(f"\nSegmentation complete.")
    print(f"FPC: {results['metrics']['fpc']:.4f}")
    print(f"Iterations: {results['metrics']['iterations']}")
    
    segmenter.save_results(results, args.output)
    print(f"Results saved to: {args.output}/")


if __name__ == '__main__':
    main()